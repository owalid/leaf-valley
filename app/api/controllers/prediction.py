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

import cv2 as cv
from PIL import Image
from plantcv import plantcv as pcv
from utils.mixins import create_response

import tensorflow as tf
from keras.models import load_model

from tqdm import tqdm
from datetime import datetime as dt
from inspect import getsourcefile

current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
current_dir = current_dir[:current_dir.rfind(os.path.sep)]
current_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, current_dir[:current_dir.rfind(os.path.sep)])

from process.deep_learning.metrics import recall_m, precision_m, f1_m
from utilities.prepare_features import prepare_features
from utilities.remove_background_functions import remove_bg
from utilities.utils import get_df, set_plants_dict, CV_NORMALIZE_TYPE

warnings.filterwarnings("ignore")


models_dict = {}

class PredictionController:
    def load_models(md, md_grp):
        if md in models_dict.keys():
            return models_dict[md]

        custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m}
        # Load DP models
        if md_grp == 'DP':
            model_path = f'../../data/models_saved/{md}.h5'
            if os.path.exists(model_path) and md not in models_dict.keys():
                print(f'Info {dt.now()} : Loading {md} model\t.....')
                models_dict[md] = load_model(model_path, custom_objects)
    
        # Load ML models
        if md_grp == 'ML':
            model_path = f'../../data/models_saved/{md}.pkl.z'
            if os.path.exists(model_path) and md not in models_dict.keys():
                print(f'Info {dt.now()} : Loading {md} model\t.....')
                models_dict[md] = joblib.load(model_path)
        
        print(f'Info {dt.now()} : Loading models done >>>>>')
        return models_dict[md]

    def get_models():
        models = {}

        for md_grp in ['ML', 'DP']:
            models_dir_path = '../../data/models_saved'
            models_dir_exist = os.path.isdir(models_dir_path)
            all_models = []
            if models_dir_exist:
                all_models = [f.split(".")[0] for f in os.listdir(models_dir_path) if f'{md_grp}_' in f and ((f.split(".")[-1] == 'h5') or ((f.split(".")[-2] == 'pkl') and (f.split(".")[-1] == 'z')))]
                all_models.sort(reverse=True)
            models[md_grp] = all_models

        
        return create_response(data={'models': models})

    def get_plants():
        data_dir = '../../data/no_augmentation'
        plants_dict = set_plants_dict(get_df(data_dir))
        return create_response(data={'plants': plants_dict})

    def get_classes():
        data_dir = '../../data/no_augmentation'
        classes = [f for f in os.listdir(data_dir) if f != 'Background_without_leaves']
        classes.sort()
        return create_response(data={'classes': classes})

    def get_ml_features(f='', path='', options_dataset={}, bgr_img=None):      
        # # Image processing
        if bgr_img is None:
            bgr_img, _, _ = pcv.readimage(os.path.join(path,f), mode='bgr')
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        data = {}
        data, _ = prepare_features(data, rgb_img, options_dataset['features'], options_dataset['should_remove_bg'], options_dataset['size_img'], 
                                   CV_NORMALIZE_TYPE[options_dataset['normalize_type']] if options_dataset['normalize_type'] else False, options_dataset['crop_img'])
 
        df = pd.DataFrame.from_dict(data)
        df.index = [f]

        return df


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

        df['species'] = df.index.to_series().apply(lambda f: f.split('___')[0])
        df['desease'] = df.index.to_series().apply(lambda f: f.split('___')[1].split('/')[0])
        df['img_num'] = df.index.to_series().apply(lambda f: f.split('(')[-1].split(')')[0])

        # prediction matching
        if len(classes) == 2:
            df['matching'] = df.apply(lambda r: ((r['prediction_label'] == 'healthy') and (r['desease'] =='healthy')) or ((r['prediction_label'] == 'not healthy') and (r['desease'] !='healthy')), axis=1)
        else:
            df['matching'] = df.apply(lambda r: (r['prediction_label'].lower() == f"{r['species']}_{r['desease']}".lower()), axis=1)
            
        return df

    def dp_predict(model, class_names, folders=[], data_dir='', img_lst=[], class_name=['']):
        if len(folders) > 0:
            img_lst = []
        for f in folders:           
            bgr_img, _, _ = pcv.readimage(os.path.join(data_dir,f), mode='bgr')
            img_lst.append(bgr_img)

        images = None
        for img in img_lst:
            img = cv.cvtColor(cv.resize(np.array(img), (256,256)), cv.COLOR_BGR2RGB)
            img = cv.resize(np.array(img), tuple(model.layers[0].get_output_at(0).get_shape().as_list()[1:-1]))
            img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            images = np.array(img[tf.newaxis, ...]) if images is None else np.concatenate((images, np.array(img[tf.newaxis, ...])), axis=0) 
        
        del img_lst

        # get prediction labels
        df = pd.DataFrame(index=folders if folders != [] else class_name)
        y = model.predict(images)
        del images

        df['prediction_label'] = [class_names[le] for le in np.argmax(y, axis=1)]
        df['proba'] = y.max(axis=1)

        df['species'] = df.index.to_series().apply(lambda f: f.split('___')[0])
        df['desease'] = df.index.to_series().apply(lambda f: f.split('___')[1].split('/')[0])
        df['img_num'] = df.index.to_series().apply(lambda f: f.split('(')[-1].split(')')[0])

        # prediction matching
        if len(class_names) == 2:
            df['matching'] = df.apply(lambda r: ((r['prediction_label'] == 'healthy') and (r['desease'] =='healthy')) or ((r['prediction_label'] == 'not healthy') and (r['desease'] !='healthy')), axis=1)
        else:
            df['matching'] = df.apply(lambda r: (r['prediction_label'].lower() == f"{r['species']}_{r['desease']}".lower()), axis=1)

        return df

    def process_images(folders, data_dir, comments, ml_df, dp_df):

        output = []
        for f in folders:
            img_dict = {}
            img_dict['img_species'] = f.split('___')[0]
            img_dict['img_desease'] = f.split('___')[1].split('/')[0]
            img_dict['img_num'] = f.split('(')[-1].split(')')[0]

            # add comment if it exits
            img_dict['comment'] = ''
            if len(comments)>0:
                cmt = [x for x in comments if (x['species'] == img_dict['img_species']) and (x['desease'] == img_dict['img_desease']) and (x['img_num'] == img_dict['img_num'])]
                if len(cmt)>0:
                    img_dict['comment'] = cmt[0]['comment']
            
            # # Image processing
            bgr_img, _, _ = pcv.readimage(os.path.join(data_dir,f), mode='bgr')
            bgr_img = cv.resize(np.array(bgr_img), (256,256))
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

            if ml_df is not None:
                df = ml_df.loc[((ml_df.species==img_dict['img_species']) & (ml_df.desease==img_dict['img_desease']) & (ml_df.img_num==img_dict['img_num']))]
                img_dict['ml_prediction'] = {
                    'class': df['prediction_label'].squeeze(),
                    'score': f'{100*float(df["proba"].squeeze()):.2f}',
                    'matching': bool(df['matching'].squeeze()),                    
                }

            if dp_df is not None:
                df = dp_df.loc[(dp_df.species==img_dict['img_species']) & (dp_df.desease==img_dict['img_desease']) & (dp_df.img_num==img_dict['img_num'])]
                img_dict['dl_prediction'] = {
                    'class': df['prediction_label'].squeeze(),
                    'score': f'{100*float(df["proba"].squeeze()):.2f}',
                    'matching': bool(df['matching'].squeeze()),                    
                }        

            output.append(img_dict)

        return output             

    def check_hacking(model):                  
        return bool(model.find('/') != -1 or \
                    model.find('\\') != -1 or \
                    model.find('..') != -1 or \
                    model.find('.') != -1) 
   
    def get_randomimag(nb_img, species, desease, ml_model, dp_model):
        global models, class_names

        if not (nb_img and species and desease and (ml_model or dp_model)):
            return create_response(data={'error': 'Incorrect date input \n Please select the correct ones !!!'}, status=500)

        data_dir = '../../data/no_augmentation'
        comment_filename = '../../data/plants_comments.json'
        df = get_df(data_dir)
        indexes = df.loc[((df.specie==species)|(species=='All'))&((df.disease==desease)|(desease=='All'))].index.tolist()
        
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
            # protect to LFI and RFI attacks
            ml_model = os.path.basename(ml_model)
            ml_model = ml_model.replace("%", '')
            if PredictionController.check_hacking(ml_model):
                return {'error': 'Incorrect model name don\'t try to hack us.'}
        
            model_path = f'../../data/models_saved/{ml_model}.pkl.z'
            if os.path.exists(model_path):
                print(f'Info {dt.now()} : Start {ml_model} model processing')
                ml_model_dict = PredictionController.load_models(ml_model, 'ML')
                df_features = pd.concat(list(tqdm(current_app._executor.map(PredictionController.get_ml_features, folders, repeat(data_dir), repeat(ml_model_dict['options_dataset'])), total=len(folders))))
                ml_df = PredictionController.ml_predict(ml_model_dict, df_features)
                print(f'Info {dt.now()} : End {ml_model} model processing')
            else:
                return {'error': f'{ml_model} model not found'}   

        # Load DP model
        if dp_model:
            # protect to LFI and RFI attacks
            dp_model = os.path.basename(dp_model)
            dp_model = dp_model.replace("%", '')
            if PredictionController.check_hacking(dp_model):
                return {'error': 'Incorrect model name don\'t try to hack us.'}
        
            model_path = f'../../data/models_saved/{dp_model}.h5'
            if os.path.exists(model_path):
                print(f'Info {dt.now()} : Start {dp_model} model processing')
                dp_model_ins = PredictionController.load_models(dp_model, 'DP')
                f = h5py.File(model_path, mode='r')
                if 'class_names' in f.attrs and len(f.attrs['class_names']) > 0:
                    class_names = f.attrs.get('class_names')
                    class_names = json.loads(class_names)
                    f.close()
                    dp_df = PredictionController.dp_predict(dp_model_ins, class_names, folders=folders, data_dir=data_dir)
                    print(f'Info {dt.now()} : End {dp_model} model processing')
                else:
                    f.close()
                    return create_response(data={'error': 'Error don\'t have classes for classification.'}, status=500)
            else:
                return {'error': f'{dp_model} model not found'}   

        # Load Images and comments if exists
        output = PredictionController.process_images(folders, data_dir, comments, ml_df if ml_model else None, dp_df if dp_model else None)

        return create_response(data={str(k): v for k, v in enumerate(output)})

    def get_selectedimag(class_name, b64File, ml_model, dp_model):
        if not (b64File and (ml_model or dp_model)):
            return create_response(data={'error': 'Incorrect date input \n Please select the correct ones !!!'}, status=500)

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
        comment_filename = '../../data/plants_comments.json'
        try:
            with open(comment_filename, "r") as js_f:
                comments = json.load(js_f)
        except:
            comments = []
            with open(comment_filename, "w") as js_f:
                json.dump(comments, js_f)
 
        # add comment if it exits
        img_dict['comment'] = ''
        if len(comments)>0 and class_name:
            cmt = [x for x in comments if (x['species'] == img_dict['img_species']) and (x['desease'] == img_dict['img_desease']) and (x['img_num'] == img_dict['img_num'])]
            if len(cmt)>0:
                img_dict['comment'] = cmt[0]['comment']
        
        # Image processing
        rgb_img = np.array(Image.open(io.BytesIO(base64.b64decode(b64File))))
        bgr_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
        _, masked_img = remove_bg(cv.resize(np.array(bgr_img), (256,256)))

        # Load ML model
        if ml_model:
            # protect to LFI and RFI attacks
            ml_model = os.path.basename(ml_model)
            ml_model = ml_model.replace("%", '')
            if PredictionController.check_hacking(ml_model):
                return {'error': 'Incorrect model name don\'t try to hack us.'}

            model_path = f'../../data/models_saved/{ml_model}.pkl.z'
            if os.path.exists(model_path):
                print(f'Info {dt.now()} : Start {ml_model} model processing')
                ml_model_dict = PredictionController.load_models(ml_model, 'ML')
                df_features = PredictionController.get_ml_features(options_dataset=ml_model_dict['options_dataset'], bgr_img=bgr_img)
                df_features.index = [class_name] if class_name else ['___/(0)']
                ml_df = PredictionController.ml_predict(ml_model_dict, df_features)
                print(f'Info {dt.now()} : End {ml_model} model proessing')
            else:
                return {'error': f'{ml_model} model not found'}   

        # Load DP model
        if dp_model:
            # protect to LFI and RFI attacks
            dp_model = os.path.basename(dp_model)
            dp_model = dp_model.replace("%", '')
            if PredictionController.check_hacking(dp_model):
                return {'error': 'Incorrect model name don\'t try to hack us.'}
        
            model_path = f'../../data/models_saved/{dp_model}.h5'
            if os.path.exists(model_path):
                print(f'Info {dt.now()} : Strat {dp_model} model processing')
                dp_model_ins = PredictionController.load_models(dp_model, 'DP')
                f = h5py.File(model_path, mode='r')
                if 'class_names' in f.attrs and len(f.attrs['class_names']) > 0:
                    class_names = f.attrs.get('class_names')
                    class_names = json.loads(class_names)
                    f.close()                   
                    dp_df = PredictionController.dp_predict(dp_model_ins, class_names, img_lst=[rgb_img], class_name=[class_name] if class_name else ['___/(0)'])
                    print(f'Info {dt.now()} : End {dp_model} model proessing')
                else:
                    f.close()
                    return create_response(data={'error': 'Error don\'t have classes for classification.'}, status=500)
            else:
                return {'error': f'{dp_model} model not found'}   

        # Get ML classification
        if ml_model:
            img_dict['ml_prediction'] = {
                'class': ml_df['prediction_label'].squeeze(),
                'score': f'{100*float(ml_df["proba"].squeeze()):.2f}',
                'matching': bool(ml_df['matching'].squeeze()),                    
            }

        # Get DP prediction
        if dp_model:
            img_dict['dl_prediction'] = {
                'class': dp_df['prediction_label'].squeeze(),
                'score': f'{100*float(dp_df["proba"].squeeze()):.2f}',
                'matching': bool(dp_df['matching'].squeeze()),                    
            }    

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
        if PredictionController.check_hacking(model_name):
            return create_response(data={'error': 'Incorrect model name don\'t try to hack us.'}, status=500)
    
        model_path = f'../../data/models_saved/{model_name}.h5'
        model_exist = os.path.exists(model_path)
        
        if model_exist:
            custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m}
            dp_model_ins = load_model(model_path, custom_objects)
            f = h5py.File(model_path, mode='r')
            if 'class_names' in f.attrs and len(f.attrs['class_names']) > 0:
                class_names = f.attrs.get('class_names')
                class_names = json.loads(class_names)
                f.close()
            else:
                f.close()
                return create_response(data={'error': 'Error don\'t have classes for classification.'}, status=500)

            model = dp_model_ins
            base64_decoded = base64.b64decode(b64img)
            image = Image.open(io.BytesIO(base64_decoded))
            image_np = np.array(image)
            _, im_withoutbg_b64 = remove_bg(image_np)
            
            # call remove background function
            if should_remove_bg:
                new_img = im_withoutbg_b64
            else:
                new_img = image_np
                
            new_img = cv.cvtColor (new_img, cv.COLOR_BGR2RGB)
            new_img = cv.resize(new_img, tuple(model.layers[0].get_output_at(0).get_shape().as_list()[1:-1]))
            new_img = cv.normalize(new_img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)            

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
        comment_filename = '../../data/plants_comments.json'

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



            