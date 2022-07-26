import os
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
from keras.applications import convnext

from tqdm import tqdm
from datetime import datetime as dt
from inspect import getsourcefile


current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
current_dir = current_dir[:current_dir.rfind(os.path.sep)]
current_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, current_dir[:current_dir.rfind(os.path.sep)])

from process.deep_learning.metrics import recall_m, precision_m, f1_m
from utilities.remove_background_functions import remove_bg
from utilities.utils import get_df, set_plants_dict, preprocess_pipeline_prediction

warnings.filterwarnings("ignore")


models_dict = {}
FLASK_ENV = os.environ.get("FLASK_ENV", "dev")
s3_module = current_app._s3_module
comment_filename = '../data/plants_comments.json' if FLASK_ENV == 'dev' else 'data/plants_comments.json'
custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m, "LayerScale": convnext.LayerScale}

class PredictionController:
    def is_production():
        return FLASK_ENV == 'production' or not PredictionController.have_s3_information()
    
    def have_s3_information():
        return s3_module.S3_ACCESS_KEY_ID and s3_module.S3_SECRET_ACCESS_KEY and s3_module.S3_BASE_ENDPOINT_URL and s3_module.S3_BUCKET_NAME
    
    def get_and_store_models(model_data, model_name):
        if model_name in models_dict.keys():
            return models_dict[model_name]
        
        if PredictionController.is_production() and models_dict.keys() > 3:
            models_dict = dict(list(models_dict.items())[-2:])
        
        models_dict[model_name] = {
            'model': model_data.model,
            'classes': model_data.classes,
            'options_datasets': model_data.options_datasets
        }
        
    def load_deeplearning_model(model_path, model_name):
        if FLASK_ENV == 'dev':
            dp_model_ins = load_model(model_path, custom_objects)
            f = h5py.File(model_path, mode='r')
            if 'class_names' in f.attrs and len(f.attrs['class_names']) > 0 and 'options_dataset' in f.attrs and len(f.attrs['options_dataset']) > 0:
                options_dataset = f.attrs.get('options_dataset')
                options_dataset = json.loads(options_dataset)
                class_names = f.attrs.get('class_names')
                class_names = json.loads(class_names)
                f.close()
            else:
                f.close()
                return None

        else:
            with tempfile.NamedTemporaryFile(mode='w+b') as f:
                s3_module.s3_client.download_fileobj(s3_module.S3_BUCKET_NAME, os.path.join(s3_module.S3_MODELS_FOLDER, model_name + '.h5', f))
                dp_model_ins = load_model(f.name, custom_objects)
                h5file = h5py.File(f.name, mode='r')
                if 'class_names' in h5file.attrs and len(h5file.attrs['class_names']) > 0 and 'options_dataset' in h5file.attrs and len(h5file.attrs['options_dataset']) > 0:
                    options_dataset = f.attrs.get('options_dataset')
                    options_dataset = json.loads(options_dataset)
                    class_names = h5file.attrs.get('class_names')
                    class_names = json.loads(class_names)
                    h5file.close()
                else:
                    h5file.close()
                    return None
                
        return dp_model_ins, options_dataset, class_names
    
    def load_ml_model(model_path, model_name):
        if FLASK_ENV == 'dev':
            return joblib.load(model_path)
        else:
           with io.BytesIO() as data:
                s3_module.s3_client.download_fileobj(s3_module.S3_BUCKET_NAME, os.path.join(s3_module.S3_MODELS_FOLDER, model_name + '.pkl.z'), data)
                data.seek(0)    # move back to the beginning after writing
                model = joblib.load(data)
                
        return model
            
    def load_models(model_name, md_grp):
        if model_name in models_dict.keys():
            return models_dict[model_name]

        if PredictionController.is_production() and models_dict.keys() >= 3:
            models_dict = dict(list(models_dict.items())[-2:])
            
        ext = '.h5' if md_grp == 'DP' else '.pkl.z'
        model_path = f'../../data/models_saved/{model_name}{ext}'
        
        if not PredictionController.is_production() and os.path.exists(model_path):
            return None
        
        if md_grp == 'DP':
            model_loaded = PredictionController.load_deeplearning_model(model_path, model_name)
            if not model_loaded:
                return None
            
            model, options_dataset, class_names = model_loaded
            models_dict[model_name] = { 'model': model, 'options_dataset': options_dataset, 'class_names': class_names }
            
        else:
            ml_model = PredictionController.load_ml_model(model_path, model_name)
            
            if not ml_model:
                return None
            models_dict[model_name] = ml_model
    
        return models_dict[model_name]

    def get_models():
        models = {}
            
        for md_grp in ['ML', 'DP']:
            if PredictionController.is_production():
                all_models = [f.split(".")[0] for f in s3_module.models_list if f'{md_grp}_' in f and ((f.split(".")[-1] == 'h5') or ((f.split(".")[-2] == 'pkl') and (f.split(".")[-1] == 'z')))]
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
        if PredictionController.is_production():
            plants_dict = set_plants_dict(s3_module.get_df_leafs())
        else:
            data_dir = '../../data/no_augmentation'
            plants_dict = set_plants_dict(get_df(data_dir))
        return create_response(data={'plants': plants_dict})

    def get_classes():
        if PredictionController.is_production():
            lst_dir = s3_module.get_folders_leafs()
        else:
            data_dir = '../../data/no_augmentation'
            lst_dir = os.listdir(data_dir)
            
        classes = [f for f in lst_dir if f != 'Background_without_leaves']
        classes.sort()
        return create_response(data={'classes': classes})

    def get_ml_features(f='', path='', options_dataset={}, bgr_img=None):      
        # Image processing
        if bgr_img is None:
            if PredictionController.is_production():
                bgr_img = s3_module.get_image_from_path(os.path.join(path, f))
            else:
                bgr_img, _, _ = pcv.readimage(os.path.join(path, f), mode='bgr')

        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        data = preprocess_pipeline_prediction(rgb_img, options_dataset)
 
        df = pd.DataFrame.from_dict(data)
        df.index = [f]

        return df

    def class_split(ldf):
        ldf['species'] = ldf.index.to_series().apply(lambda f: f.split('___')[0])
        ldf['desease'] = ldf.index.to_series().apply(lambda f: f.split('___')[1].split('/')[0])
        ldf['img_num'] = ldf.index.to_series().apply(lambda f: f.split('(')[-1].split(')')[0])

    def ml_predict(model_dict, df_features):       
        # Get ML model components
        model   = model_dict['ml_model']
        le      = model_dict['ml_label_encoder']
        md_feat = model_dict['ml_features']
        scaler  = model_dict['ml_scaler']
        
        classes = list(le.classes_)

        # Data Normalization
        df_features[md_feat] = scaler.transform(df_features[md_feat]).astype(np.float32)

        df = pd.DataFrame(index=df_features.index)
        # get prediction labels
        df['preds'] = model.predict(df_features[md_feat])
        df['prediction_label'] = le.inverse_transform(df['preds'])
        df['proba'] = model.predict_proba(df_features[md_feat]).max(axis=1).tolist()

        # Split the class name
        PredictionController.class_split(df)

        # prediction matching
        if len(classes) == 2:
            df['matching'] = df.apply(lambda r: ((r['prediction_label'] == 'healthy') and (r['desease'] =='healthy')) or ((r['prediction_label'] == 'not healthy') and (r['desease'] !='healthy')), axis=1)
        else:
            df['matching'] = df.apply(lambda r: (r['prediction_label'].lower() == f"{r['species']}_{r['desease']}".lower()), axis=1)
            
        return df

    def dp_predict(model, class_names, options_dataset, folders=[], data_dir='', img_lst=[], class_name=['']):
        if len(folders) > 0:
            img_lst = []
            
        for f in folders:
            if PredictionController.is_production():
                bgr_img = s3_module.get_image_from_path(f)
            else:
                bgr_img, _, _ = pcv.readimage(os.path.join(data_dir,f), mode='bgr')

            img_lst.append(bgr_img)

        images = None
        for img in img_lst:
            img = preprocess_pipeline_prediction(img, options_dataset)
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
        if df is not None:
            df = df.loc[((df.species==dict['img_species']) & (df.desease==dict['img_desease']) & (df.img_num==dict['img_num']))]
            dict[key] = {
                'class': df['prediction_label'].squeeze(),
                'score': f'{100*float(df["proba"].squeeze()):.2f}',
                'matching': bool(df['matching'].squeeze()),                    
            }        

    def process_images(folders, data_dir, comments, ml_df, dp_df):

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
                bgr_img = s3_module.get_image_from_path(f)
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

            # Get DP prediction output
            PredictionController.get_prediction_output(dp_df, img_dict, 'dp_prediction')

            output.append(img_dict)

        return output             

    def check_lfi_attack(model):
        # protect to LFI and RFI attacks
        model = os.path.basename(model)
        model = model.replace("%", '')                       
        return bool(model.find('/') != -1 or \
                    model.find('\\') != -1 or \
                    model.find('..') != -1 or \
                    model.find('.') != -1) 
   
    def get_randomimag(nb_img, species, desease, ml_model, dp_model):
        global models, class_names

        if not (nb_img and species and desease and (ml_model or dp_model)):
            return create_response(data={'error': 'Incorrect date input \n Please select the correct ones !!!'}, status=500)

        data_dir = '../../data/no_augmentation'
        
        if PredictionController.is_production():
            df = s3_module.get_df_leafs()
        else:
            df = get_df(data_dir)
        indexes = df.loc[((df.specie==species)|(species=='All'))&((df.disease==desease)|(desease=='All'))].index.tolist()
        
        if PredictionController.is_production():
            folders = s3_module.files_leafs
        else:
            folders = []
            for idx in indexes:
                folders.append([os.path.join(idx,f) for f in os.listdir(os.path.join(data_dir,idx))])
        
        folders = sum(folders,[])
        random.shuffle(folders)
        folders = random.sample(folders, nb_img)

        # open comment file
        try:
            with open(comment_filename, "r") as js_f:
                comments = json.load(js_f)
        except:
            comments = []
            with open(comment_filename, "w") as js_f:
                json.dump(comments, js_f)

        # ML model Processing
        if ml_model:
            if PredictionController.check_lfi_attack(ml_model):
                return {'error': 'Incorrect model name.'}
        
            ml_model_dict = PredictionController.load_models(ml_model, 'ML')
            if ml_model_dict:
                df_features = pd.concat(list(tqdm(current_app._executor.map(PredictionController.get_ml_features, folders, repeat(data_dir), repeat(ml_model_dict['options_dataset'])), total=len(folders))))
                ml_df = PredictionController.ml_predict(ml_model_dict, df_features)
                print(f'Info {dt.now()} : End {ml_model} model processing')
            else:
                return {'error': f'{ml_model} model not found'}   

        # Load DP model
        if dp_model:
            if PredictionController.check_lfi_attack(dp_model):
                return {'error': 'Incorrect model name.'}
        
            data_model_loaded = PredictionController.load_models(dp_model, 'DP')
            
            if data_model_loaded:
                dp_model_instance, options_dataset, class_names = data_model_loaded
                dp_df = PredictionController.dp_predict(dp_model_instance, class_names, options_dataset, folders=folders, data_dir=data_dir)
            else:
                return {'error': f'{dp_model} model not found'}   

        # Load Images and comments if exists
        output = PredictionController.process_images(folders, data_dir, comments, ml_df if ml_model else None, dp_df if dp_model else None)

        return create_response(data={str(k): v for k, v in enumerate(output)})

    def get_selectedimage(class_name, b64File, ml_model, dp_model):
        if not (b64File and (ml_model or dp_model)):
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
            with open(comment_filename, "r") as js_f:
                comments = json.load(js_f)
        except:
            comments = []
            with open(comment_filename, "w") as js_f:
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
                return {'error': 'Incorrect model name.'}

            ml_model_dict = PredictionController.load_models(ml_model, 'ML')
            
            if ml_model_dict:
                df_features = PredictionController.get_ml_features(options_dataset=ml_model_dict['options_dataset'], bgr_img=bgr_img)
                df_features.index = [class_name] if class_name else ['___/(0)']
                ml_df = PredictionController.ml_predict(ml_model_dict, df_features)
                PredictionController.get_prediction_output(ml_df, img_dict)
            else:
                return {'error': f'{ml_model} model not found'}   

        # Load DP model
        if dp_model:
            if PredictionController.check_lfi_attack(dp_model):
                return {'error': 'Incorrect model name.'}
        
            data_model_loaded = PredictionController.load_models(dp_model, 'DP')
            
            if data_model_loaded:
                print(f'Info {dt.now()} : Strat {dp_model} model processing')
                dp_model_instance, options_dataset, class_names = data_model_loaded
                
                dp_df = PredictionController.dp_predict(dp_model_instance, class_names, options_dataset, img_lst=[rgb_img], class_name=[class_name] if class_name else ['___/(0)'])
                PredictionController.get_prediction_output(dp_df, img_dict)
            else:
                return {'error': f'{dp_model} model not found'}   

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
        model_name = os.path.basename(model_name)
        model_name = model_name.replace("%", '')
        if PredictionController.check_lfi_attack(model_name):
            return create_response(data={'error': 'Incorrect model name.'}, status=500)
    
        if FLASK_ENV == 'dev' or not PredictionController.have_s3_information:
            model_path = f'../../data/models_saved/{model_name}.h5'
            model_exist = os.path.exists(model_path)
        else:
            models_availables = PredictionController.get_s3_models()
            model_exist = model_name in models_availables
        
        if model_exist:
            if not model_name in models.keys():
                data_model_loaded = PredictionController.load_deeplearning_model(model_path, model_name)
                if not data_model_loaded:
                    return create_response(data={'error': 'Error don\'t have classes for classification.'}, status=500)
                
                model, options_dataset, class_names = data_model_loaded
                        
            base64_decoded = base64.b64decode(b64img)
            image = Image.open(io.BytesIO(base64_decoded))
            image_np = np.array(image)
            _, im_withoutbg_b64 = remove_bg(image_np)
            
            # call remove background function
            if should_remove_bg:
                new_img = im_withoutbg_b64
            else:
                new_img = image_np
                
            new_img = preprocess_pipeline_prediction(new_img, options_dataset)

            # get prediction labels
            y = model.predict(new_img[tf.newaxis, ...])
            label_encoded = np.argmax(y, axis=-1)[0]
            prediction_label = class_names[label_encoded]
            prediction_accuracy = str(y.max())

            # convert numpy array image to base64
            _, img_arr = cv.imencode('.jpg', im_withoutbg_b64)
            im_withoutbg_b64 = base64.b64encode(img_arr).decode('utf-8')
            prediction_data = {
                'prediction': prediction_label,
                'score': str(prediction_accuracy),
                'im_withoutbg_b64': im_withoutbg_b64
            }
            return create_response(data=prediction_data)
        else:
            return create_response(data={'error': f'{model_name} model not found'}, status=500)
            
    def process_comment(method, comment):

        with open(comment_filename, "r") as js_f:
            try:
                comments = json.load(js_f)
            except:
                comments = []

        # insert comment
        if method == 'insert':
            comments.append(comment)
        if method == 'update':
            comments = [x for x in comments if not ((x['species'] == comment['species']) and (x['desease'] == comment['desease']) and (x['img_num'] == comment['img_num']))]
            comments.append(comment)
        if method == 'delete':
            comments = [x for x in comments if not ((x['species'] == comment['species']) and (x['desease'] == comment['desease']) and (x['img_num'] == comment['img_num']))]

        # Save comments file
        with open(comment_filename, "w") as js_f:
            json.dump(comments, js_f, indent = 2)

        return create_response(data={'result': f'{method} with success'})
