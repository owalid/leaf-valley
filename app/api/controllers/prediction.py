import os
import re
import io
import sys
import h5py
import json
import base64
import random
import joblib
import warnings
import numpy as np
import pandas as pd
from itertools import repeat
from flask import current_app
import tempfile

import cv2 as cv
from PIL import Image
from plantcv import plantcv as pcv
from utils.mixins import create_response

import tensorflow as tf
from keras.models import load_model
from flask import g
from tqdm import tqdm
from datetime import datetime as dt
from inspect import getsourcefile

from modules.s3_module import S3Module

current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
current_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, current_dir[:current_dir.rfind(os.path.sep)])

from process.deep_learning.metrics import recall_m, precision_m, f1_m, LayerScale
from utilities.remove_background_functions import remove_bg
from utilities.prepare_features import prepare_features
from utilities.utils import set_plants_dict, get_df, safe_get_item, safe_open_w
from utilities.image_transformation import rgbtobgr


class PredictionController:
    models_dict = {}
    s3_module = S3Module()
    FLASK_ENV = os.environ.get("FLASK_ENV", "dev")
    comment_filename = 'data/plants_comments.json' if FLASK_ENV == 'prod' else '../data/plants_comments.json'
    custom_objects = { 'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m, "LayerScale": LayerScale }
    
    def preprocess_pipeline_prediction(rgb_img, options, is_deep_learning_model=False):
        '''
            Description: Image preprocess pipeline for prediction
            
            Parameters:
                - rgb_img: image in rgb format (np.ndarray)
                - options: options for preprocess pipeline (dict)
                - is_deep_learning_model: is deep learning model (bool)
        '''

        normalize_type = safe_get_item(options, 'normalize_type', None)
        data = {}
        img, _ = prepare_features(data, rgb_img, safe_get_item(options,'features',{}), safe_get_item(options, 'should_remove_bg'),
                                size_img=safe_get_item(options, 'size_img', None),\
                                normalize_type=normalize_type,\
                                crop_img=safe_get_item(options, 'crop_img', False),\
                                is_deep_learning_features=safe_get_item(options, 'is_deep_learning_features', False) or is_deep_learning_model)
            
        return img
        
    def is_production():
        '''
            Description: Check if the application is in production mode and have s3 information
        '''
        
        return PredictionController.FLASK_ENV == 'prod' or not PredictionController.have_s3_information()
    
    def have_s3_information():
        '''
            Description: Check if s3 information is available
        '''
        
        return PredictionController.s3_module.S3_ACCESS_KEY_ID and PredictionController.s3_module.S3_SECRET_ACCESS_KEY and PredictionController.s3_module.S3_BASE_ENDPOINT_URL and PredictionController.s3_module.S3_BUCKET_NAME
        
    def load_deeplearning_model(model_path, model_name):
        '''
            Description: Load deeplearning model from s3 or local according to the environment
            
            Parameters:
                - model_path: path to model file (str)
                - model_name: name of model (str)
        '''
        
        if PredictionController.is_production():
            with tempfile.NamedTemporaryFile(mode='w+b') as f:
                print("PredictionController.s3_module.S3_MODELS_FOLDER", PredictionController.s3_module.S3_MODELS_FOLDER)
                print("model_name", model_name)
                path_model = os.path.join(PredictionController.s3_module.S3_MODELS_FOLDER, model_name + '.h5')
                print("path_model", path_model)
                PredictionController.s3_module.s3_client.download_fileobj(PredictionController.s3_module.S3_BUCKET_NAME, path_model, f)
                dl_model_ins = load_model(f.name, PredictionController.custom_objects)
                print("[+] dl_model_ins", type(dl_model_ins))
                h5file = h5py.File(f.name, mode='r')
                print("[+] h5file.attrs", h5file.attrs.keys())
                h5file_keys = list(h5file.attrs.keys())
                print("[+] h5file_keys", h5file_keys)
                if 'class_names' in h5file_keys and len(h5file.attrs['class_names']) > 0 and 'options_dataset' in h5file_keys and len(h5file.attrs['options_dataset']) > 0:
                    options_dataset = h5file.attrs.get('options_dataset')
                    options_dataset = json.loads(options_dataset)
                    class_names = h5file.attrs.get('class_names')
                    class_names = json.loads(class_names)
                    h5file.close()
                else:
                    h5file.close()
                    return None

        else:
            dl_model_ins = load_model(model_path, PredictionController.custom_objects)
            f = h5py.File(model_path, mode='r')
            h5file_keys = list(f.attrs.keys())
            if 'class_names' in  h5file_keys and len(f.attrs['class_names']) > 0 and 'options_dataset' in h5file_keys and len(f.attrs['options_dataset']) > 0:
                options_dataset = f.attrs.get('options_dataset')
                options_dataset = json.loads(options_dataset)
                class_names = f.attrs.get('class_names')
                class_names = json.loads(class_names)
                f.close()
            else:
                f.close()
                return None
                
        return dl_model_ins, options_dataset, class_names
    
    def load_ml_model(model_path, model_name):
        '''
            Description: Load model from s3 or local according to the environment
            
            Parameters:
                - model_path: path to model file (str)
                - model_name: name of model (str)
        '''
        
        if PredictionController.FLASK_ENV == 'dev':
            return joblib.load(model_path)
        else:
           with io.BytesIO() as data:
                PredictionController.s3_module.s3_client.download_fileobj(PredictionController.s3_module.S3_BUCKET_NAME, os.path.join(PredictionController.s3_module.S3_MODELS_FOLDER, model_name + '.pkl.z'), data)
                data.seek(0)    # move back to the beginning after writing
                model = joblib.load(data)
                
        return model
            
    def load_models(model_name, md_grp):
        '''
            Description: Load models from S3 or local according to the environment and md_grp ('ML' or 'DL')
            
            Parameters: 
                - model_name: name of the model (str)
                - md_grp: group of the model in the config file (dict)
        '''
        
        if model_name in PredictionController.models_dict.keys():
            return PredictionController.models_dict[model_name]

        if PredictionController.is_production() and len(PredictionController.models_dict.keys()) >= 3:
            PredictionController.models_dict = dict(list(PredictionController.models_dict.items())[-2:])
            
        ext = '.h5' if md_grp == 'DL' else '.pkl.z'
        model_path = f'../../data/models_saved/{model_name}{ext}'
        
        if not PredictionController.is_production() and not os.path.exists(model_path):
            return None
        
        if md_grp == 'DL':
            model_loaded = PredictionController.load_deeplearning_model(model_path, model_name)
            if not model_loaded:
                return None
            
            model, options_dataset, class_names = model_loaded
            PredictionController.models_dict[model_name] = { 'model': model, 'options_dataset': options_dataset, 'class_names': class_names }
            
        else:
            ml_model = PredictionController.load_ml_model(model_path, model_name)
            
            if not ml_model:
                return None
            PredictionController.models_dict[model_name] = ml_model
    
        return PredictionController.models_dict[model_name]

    def get_models():
        '''
            Route: GET /api/models/
            
            Description: Get alls models available
        '''
        
        models = {}
            
        for md_grp in ['ML', 'DL']:
            if PredictionController.is_production():
                all_models = [f.split(".")[0] for f in PredictionController.s3_module.models_list if f'{md_grp}_' in f and ((f.split(".")[-1] == 'h5') or ((f.split(".")[-2] == 'pkl') and (f.split(".")[-1] == 'z')))]
                all_models.sort(reverse=True)
                models[md_grp] = all_models
            else:
                models_dir_path = '../../data/models_saved'
                models_dir_exist = os.path.isdir(models_dir_path)
                all_models = []
                if models_dir_exist:
                    all_models = [f.split(".")[0] for f in os.listdir(models_dir_path) if f'{md_grp}_' in f and ((f.split(".")[-1] == 'h5') or ((f.split(".")[-2] == 'pkl') and (f.split(".")[-1] == 'z')))]
                    all_models.sort(reverse=True)
                models[md_grp] = all_models
                
        return create_response(data={'models': models})

    def get_plants():
        '''
            Route: GET /api/models/plants
            
            Description: Get all plants names.
        '''
        
        if PredictionController.is_production():
            plants_dict = set_plants_dict(PredictionController.s3_module.get_df_leafs())
        else:
            data_dir = '../../data/no_augmentation'
            plants_dict = set_plants_dict(get_df(data_dir))
        return create_response(data={'plants': plants_dict})

    def get_classes():
        '''
            Route: GET /api/models/classes
            
            Description: Get the classes (folders leaves)
        '''
        
        if PredictionController.is_production():
            lst_dir = PredictionController.s3_module.get_folders_leafs()
        else:
            data_dir = '../../data/no_augmentation'
            lst_dir = os.listdir(data_dir)
            
        classes = [f for f in lst_dir if f != 'Background_without_leaves']
        classes.sort()
        return create_response(data={'classes': classes})

    def get_ml_features(f='', path='', options_dataset={}, bgr_img=None):
        '''
            Description: Utility function to get the features of a given image for machine learning models
            
            Parameters:
                f: name of the image (str)
                path: path of the image (str)
                options_dataset: options dataset of the model (dict)
                bgr_img: image in BGR format (np.array)
        '''
        print("test")
        # Image processing
        if bgr_img is None:
            if PredictionController.is_production():
                bgr_img = PredictionController.s3_module.get_image_from_path(os.path.join(path, f))
            else:
                bgr_img, _, _ = pcv.readimage(os.path.join(path, f), mode='bgr')

        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        data = PredictionController.preprocess_pipeline_prediction(rgb_img, options_dataset)
 
        df = pd.DataFrame.from_dict(data)
        df.index = [f]

        return df

    def class_split(ldf):
        ldf['species'] = ldf.index.to_series().apply(lambda f: f.split('___')[0])
        ldf['desease'] = ldf.index.to_series().apply(lambda f: f.split('___')[1].split('/')[0])
        ldf['img_num'] = ldf.index.to_series().apply(lambda f: f.split('(')[-1].split(')')[0])

    def ml_predict(model_dict, df_features):
        '''
            Description: Utility function to predict the class of an image with a trained machine learning model
            
            Parameters:
                - model_dict: dictionary with the model data to use and its options (dict)
                - df_features: dataframe with the features of the image to predict (pd.DataFrame)
        '''
        # Get ML model components
        model = model_dict['ml_model']
        le = model_dict['ml_label_encoder']
        model_feature = model_dict['ml_features']
        scaler = model_dict['ml_scaler']
        
        classes = list(le.classes_)

        # Data Normalization
        df_features[model_feature] = scaler.transform(df_features[model_feature]).astype(np.float32)

        df = pd.DataFrame(index=df_features.index)
        # get prediction labels
        df['preds'] = model.predict(df_features[model_feature])
        df['prediction_label'] = le.inverse_transform(df['preds'])
        df['proba'] = model.predict_proba(df_features[model_feature]).max(axis=1).tolist()

        # Split the class name
        PredictionController.class_split(df)

        # prediction matching
        if len(classes) == 2:
            df['matching'] = df.apply(lambda r: ((r['prediction_label'] == 'healthy') and (r['desease'] =='healthy')) or ((r['prediction_label'] == 'not healthy') and (r['desease'] !='healthy')), axis=1)
        else:
            df['matching'] = df.apply(lambda r: (r['prediction_label'].lower() == f"{r['species']}_{r['desease']}".lower()), axis=1)
            
        return df

    def dl_predict(model, class_names, options_dataset, folders=[], data_dir='', img_lst=[], class_name=['']):
        '''
            Description: Utility function to launch prediction with Deep Learning model
            
            Parameters:
                - model: Deep Learning model (keras.models.Model)
                - class_names: list of class names (list)
                - options_dataset: options dataset (dict)
                - folders: list of folders (list)
                - data_dir: data directory (str)
                - img_lst: list of images (list)
                - class_name: class name (str)
        '''
        
        if len(folders) > 0:
            img_lst = []
            
        for f in folders:
            if PredictionController.is_production():
                bgr_img = PredictionController.s3_module.get_image_from_path(f)
            else:
                bgr_img, _, _ = pcv.readimage(os.path.join(data_dir,f), mode='bgr')

            img_lst.append(bgr_img)

        images = None
        for img in img_lst:
            img = PredictionController.preprocess_pipeline_prediction(img, options_dataset, is_deep_learning_model=True)
            images = np.array(img[tf.newaxis, ...]) if images is None else np.concatenate((images, np.array(img[tf.newaxis, ...])), axis=0)
        
        del img_lst

        # get prediction labels
        df = pd.DataFrame(index=folders if folders != [] else class_name)
        y = model.predict(images)
        del images

        df['prediction_label'] = [class_names[le] for le in np.argmax(y, axis=1)]
        df['proba'] = y.max(axis=1)

        # Split the class name
        PredictionController.class_split(df)

        # prediction matching
        if len(class_names) == 2:
            df['matching'] = df.apply(lambda r: ((r['prediction_label'] == 'healthy') and (r['desease'] =='healthy')) or ((r['prediction_label'] == 'not healthy') and (r['desease'] !='healthy')), axis=1)
        else:
            df['matching'] = df.apply(lambda r: (r['prediction_label'].lower() == f"{r['species']}_{r['desease']}".lower()), axis=1)

        return df

    def get_prediction_output(df, dict, key):
        '''
         Description: Utilitiy function to create the output for the prediction
         
         Parameters: 
            - df: DataFrame with the prediction results. (pd.DataFrame)
            - dict: Dictionary result with the prediction results, with two keys: 'ML' and 'DL'. (dict)
            - key: String with the key to access the dictionary. (str)
        '''
        
        if df is not None:
            df = df.loc[((df.species==dict['img_species']) & (df.desease==dict['img_desease']) & (df.img_num==dict['img_num']))]
            dict[key] = {
                'class': df['prediction_label'].squeeze(),
                'score': f'{100*float(df["proba"].squeeze()):.2f}',
                'matching': bool(df['matching'].squeeze()),                    
            }  

    def process_images(folders, data_dir, comments, ml_df, dl_df):

        output = []
        for f in folders:
            img_dict = {}
            img_dict['img_species'] = f.split('___')[0]
            img_dict['img_desease'] = f.split('___')[1].split('/')[0]
            img_dict['img_num'] = f.split('(')[-1].split(')')[0]

            # add comment if it exits
            img_dict['comment'] = ''
            if len(comments) > 0:
                cmt = [x for x in comments if (x['species'] == img_dict['img_species']) and (x['desease'] == img_dict['img_desease']) and (x['img_num'] == img_dict['img_num'])]
                if len(cmt) > 0:
                    img_dict['comment'] = cmt[0]['comment']
            
            # # Image processing
            if PredictionController.is_production():
                bgr_img = PredictionController.s3_module.get_image_from_path(f)
            else:
                bgr_img, _, _ = pcv.readimage(os.path.join(data_dir,f), mode='bgr')

            bgr_img = cv.resize(np.array(bgr_img), (256, 256))
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            _, masked_img = remove_bg(bgr_img)

            # convert numpy array image to base64
            _, rgb_img = cv.imencode('.jpg', bgr_img)
            rgb_img = base64.b64encode(rgb_img).decode('utf-8')
            _, masked_img = cv.imencode('.jpg', masked_img)
            masked_img = base64.b64encode(masked_img).decode('utf-8')

            # add images to the dictionary
            img_dict['rgb_img'] = rgb_img
            img_dict['masked_img'] = masked_img

            del bgr_img, rgb_img, masked_img

            # Get ML prediction output
            PredictionController.get_prediction_output(ml_df, img_dict, 'ml_prediction')

            # Get DL prediction output
            PredictionController.get_prediction_output(dl_df, img_dict, 'dl_prediction')

            output.append(img_dict)

        return output             

    def check_lfi_attack(model_name):
        '''
            Utilities function to check if the model_name string contain LFI attack
            
            Parameters:
                - model_name: model name (string)
        '''
        # protect to LFI and RFI attacks
        model_name = os.path.basename(model_name)
        model_name = model_name.replace("%", '')                       
        return bool(model_name.find('/') != -1 or \
                    model_name.find('\\') != -1 or \
                    model_name.find('..') != -1 or \
                    model_name.find('.') != -1) 
   
    def get_randomimage(nb_img, species, desease, ml_model, dl_model):
        '''
         Route: POST /api/models/random-img
         
         Description: Run predictions on dataset with random images and return the results.
         
         Parameters:
            - nb_img: number of images to predict (int)
            - species: species name (string)
            - desease: desease name (string)
            - ml_model: machine learning model name (string)
            - dl_model: deep learning model name (string)
        '''
        global models, class_names

        if not (nb_img and species and desease and (ml_model or dl_model)):
            return create_response(data={'error': 'Incorrect date input \n Please select the correct ones !!!'}, status=500)

        data_dir = '../../data/no_augmentation'
        
        if PredictionController.is_production():
            df = PredictionController.s3_module.get_df_leafs()
        else:
            df = get_df(data_dir)
        indexes = df.loc[((df.specie==species)|(species=='All'))&((df.disease==desease)|(desease=='All'))].index.tolist()
        
        if PredictionController.is_production():
            folders = []
            all_folder = PredictionController.s3_module.get_folders_leafs()
            for name_folder in all_folder:
                folders.append([f for f in PredictionController.s3_module.files_leafs if name_folder in f])

        else:
            folders = []
            for idx in indexes:
                folders.append([os.path.join(idx,f) for f in os.listdir(os.path.join(data_dir,idx))])

        folders = sum(folders,[])
        random.shuffle(folders)

        folders = random.sample(folders, nb_img)
        # open comment file
        try:
            with safe_open_w(PredictionController.comment_filename, option_open="r") as js_f:
                comments = json.load(js_f)
        except:
            comments = []
            with safe_open_w(PredictionController.comment_filename, option_open="w") as js_f:
                json.dump(comments, js_f)

        # ML model Processing
        if ml_model:
            if PredictionController.check_lfi_attack(ml_model):
                return create_response(data={'error': 'Incorrect model name.'}, status=500)
        
            ml_model_dict = PredictionController.load_models(ml_model, 'ML')
            if ml_model_dict:
                df_features = pd.concat(list(tqdm(current_app._executor.map(PredictionController.get_ml_features, folders, repeat(data_dir), repeat(ml_model_dict['options_dataset'])), total=len(folders))))
                ml_df = PredictionController.ml_predict(ml_model_dict, df_features)
            else:
                return create_response(data={'error': f'{ml_model} model not found'}, status=500)

        # Load DL model
        if dl_model:
            if PredictionController.check_lfi_attack(dl_model):
                return create_response(data={'error': 'Incorrect model name.'}, status=500)
        
            data_model_loaded = PredictionController.load_models(dl_model, 'DL')
            
            if data_model_loaded:
                dl_df = PredictionController.dl_predict(data_model_loaded['model'], data_model_loaded['class_names'], data_model_loaded['options_dataset'], folders=folders, data_dir=data_dir)
            else:
                return create_response(data={'error': f'{dl_model} model not found'}, status=500)

        # Load Images and comments if exists
        output = PredictionController.process_images(folders, data_dir, comments, ml_df if ml_model else None, dl_df if dl_model else None)

        return create_response(data={str(k): v for k, v in enumerate(output)})

    def get_selectedimage(class_name, b64File, ml_model, dl_model):
        if not (b64File and (ml_model or dl_model)):
            return create_response(data={'error': 'Incorrect data input \n Please select the correct ones !!!'}, status=500)

        img_dict = {}
        if class_name:
            img_dict['img_species'] = class_name.split('___')[0]
            img_dict['img_desease'] = class_name.split('___')[1].split('/')[0]
            img_dict['img_num'] = class_name.split('(')[-1].split(')')[0]
        else:
            img_dict['img_species'] = ''
            img_dict['img_desease'] = ''
            img_dict['img_num'] = None

        # open comment file
        try:
            with safe_open_w(PredictionController.comment_filename, option_open="r") as js_f:
                comments = json.load(js_f)
        except:
            comments = []
            with safe_open_w(PredictionController.comment_filename, option_open="w") as js_f:
                json.dump(comments, js_f)
 
        # add comment if it exits
        img_dict['comment'] = ''
        if len(comments) > 0 and class_name:
            cmt = [x for x in comments if (x['species'] == img_dict['img_species']) and (x['desease'] == img_dict['img_desease']) and (x['img_num'] == img_dict['img_num'])]
            if len(cmt) > 0:
                img_dict['comment'] = cmt[0]['comment']
        
        # Image processing
        rgb_img = np.array(Image.open(io.BytesIO(base64.b64decode(b64File))))
        bgr_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
        _, masked_img = remove_bg(cv.resize(np.array(bgr_img), (256, 256)))

        # Load ML model
        if ml_model:
            if PredictionController.check_lfi_attack(ml_model):
                return create_response(data={'error': 'Incorrect model name.'}, status=500)

            ml_model_dict = PredictionController.load_models(ml_model, 'ML')
            
            if ml_model_dict:
                df_features = PredictionController.get_ml_features(options_dataset=ml_model_dict['options_dataset'], bgr_img=bgr_img)
                df_features.index = [class_name] if class_name else ['___/(0)']
                ml_df = PredictionController.ml_predict(ml_model_dict, df_features)
                PredictionController.get_prediction_output(ml_df, img_dict, 'ML')
            else:
                return create_response(data={'error': f'{ml_model} model not found'}, status=500)

        # Load DL model
        if dl_model:
            if PredictionController.check_lfi_attack(dl_model):
                return create_response(data={'error': 'Incorrect model name.'}, status=500)
        
            data_model_loaded = PredictionController.load_models(dl_model, 'DL')
            
            if data_model_loaded:
                dl_df = PredictionController.dl_predict(data_model_loaded['model'], data_model_loaded['class_names'], data_model_loaded['options_dataset'], img_lst=[rgb_img], class_name=[class_name] if class_name else ['___/(0)'])
                PredictionController.get_prediction_output(dl_df, img_dict, 'DL')
            else:
                return create_response(data={'error': f'{dl_model} model not found'}, status=500)

        # convert numpy array image to base64
        _, rgb_img = cv.imencode('.jpg', bgr_img)
        rgb_img = base64.b64encode(rgb_img).decode('utf-8')
        _, masked_img = cv.imencode('.jpg', masked_img)
        masked_img = base64.b64encode(masked_img).decode('utf-8')

        # add images to the dictionary
        img_dict['rgb_img'] = rgb_img
        img_dict['masked_img'] = masked_img

        del bgr_img, rgb_img, masked_img

        return create_response(data={str(k): v for k, v in enumerate([img_dict])})

    def predict(b64img, model_name, should_remove_bg):
        
        # protect to LFI and RFI attacks
        if PredictionController.check_lfi_attack(model_name):
            return create_response(data={'error': 'Incorrect model name.'}, status=500)
        
        type_model = 'DL' if "DL_" in model_name else 'ML'
        model_data = PredictionController.load_models(model_name, type_model)
        
        if not model_data:
            return create_response(data={'error': 'Error don\'t have classes for classification.'}, status=500)
        
        base64_decoded = base64.b64decode(b64img)
        rgb_img = Image.open(io.BytesIO(base64_decoded))
        rgb_img = np.array(rgb_img)
        options_dataset = model_data['options_dataset']
        
        if type_model == 'ML':
            index = '___/(0)'
            bgr_img = rgbtobgr(rgb_img)
            df_features = PredictionController.get_ml_features(options_dataset=options_dataset, bgr_img=bgr_img)
            df_features.index = [index]
            ml_df = PredictionController.ml_predict(model_data, df_features)
            
            prediction_label = ml_df['prediction_label'][index]
            prediction_score = ml_df['proba'][index]
        else:
            model = model_data['model']
            class_names = model_data['class_names']    
        
            new_img = PredictionController.preprocess_pipeline_prediction(rgb_img, options_dataset, is_deep_learning_model=True)
            y = model.predict(new_img[tf.newaxis, ...])
            label_encoded = np.argmax(y, axis=-1)[0]
            prediction_label = class_names[label_encoded]
            prediction_score = str(y.max())

        if should_remove_bg:
            _, rgb_img = remove_bg(rgb_img)

        # convert numpy array image to base64
        _, img_arr = cv.imencode('.jpg', rgb_img)
        im_withoutbg_b64 = base64.b64encode(img_arr).decode('utf-8')
        prediction_data = {
            'prediction': prediction_label,
            'score': f'{100*float(prediction_score):.2f}',
            'im_withoutbg_b64': im_withoutbg_b64
        }
        return create_response(data=prediction_data)
            

