import os
from flask import current_app
import io
import sys
import h5py
import json
import base64
import random
import joblib
import numpy as np
import pandas as pd
import warnings
import concurrent.futures
from itertools import repeat
import parmap

import cv2 as cv
from PIL import Image, ImageEnhance
from plantcv import plantcv as pcv
from utils.mixins import create_response, serialize_list
# from tensorflow import keras

import tensorflow as tf
from keras.models import load_model

from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
current_dir = current_dir[:current_dir.rfind(path.sep)]
current_dir = current_dir[:current_dir.rfind(path.sep)]
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from process.deep_learning.metrics import recall_m, precision_m, f1_m
from utilities.utils import get_df, set_plants_dict, update_data_dict

from utilities.remove_background_functions import remove_bg
from utilities.extract_features import get_pyfeats_features, get_lpb_histogram, get_hue_moment, get_haralick, get_hsv_histogram, get_lab_histogram, get_graycoprops, get_lab_img, get_hsv_img

from datetime import datetime as dt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# from flask import jsonify

class_names = None
models = {}

class PredictionController:
    def get_models(md_grp):
        models_dir_path = '../../data/models_saved'
        models_dir_exist = os.path.isdir(models_dir_path)
        all_models = []
        if models_dir_exist:
            all_models = [f.split(".")[-2] for f in os.listdir(models_dir_path) if f'{md_grp}_' in f and ((f.split(".")[-1] == 'h5') or (f.split(".")[-1] == 'joblib'))]
            all_models.sort()
        return create_response(data={'models': all_models})

    def get_plants():
        data_dir = '../../data/no_augmentation'
        plants_dict = set_plants_dict(get_df(data_dir))
        return create_response(data={'plants': plants_dict})

    def get_classes():
        data_dir = '../../data/no_augmentation'
        classes = [f for f in os.listdir(data_dir) if f != 'Background_without_leaves']
        classes.sort()
        return create_response(data={'classes': classes})


    def generate_img_without_bg(mask, raw_img):
        im = Image.fromarray(raw_img)
        enhancer = ImageEnhance.Sharpness(im)
        pill_img = enhancer.enhance(2)
        masked_img = np.array(pill_img)
        masked_img = cv.resize(masked_img, (256,256))
        
        return masked_img

    def get_ml_features(f, path):      
        # # Image processing
        bgr_img, _, _ = pcv.readimage(os.path.join(path,f), mode='bgr')
        bgr_img = cv.resize(np.array(bgr_img), (256,256))
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        bgr_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        mask, _ = remove_bg(bgr_img)
        masked_img = PredictionController.generate_img_without_bg(mask, bgr_img)

        # FEATURES MACHINE LEARNING
        data = {}
        features = get_graycoprops(masked_img)
        for feature in features:
            data = update_data_dict(data, feature, features[feature])
        features = get_lpb_histogram(masked_img)
        for feature in features:
            data = update_data_dict(data, feature, features[feature])
        features = get_hue_moment(masked_img)
        for feature in features:
            data = update_data_dict(data, feature, features[feature])
        features = get_haralick(masked_img)
        for feature in features:
            data = update_data_dict(data, feature, features[feature])
        features = get_hsv_histogram(masked_img)
        for feature in features:
            data = update_data_dict(data, feature, features[feature])
        features = get_lab_histogram(masked_img)
        for feature in features:
            data = update_data_dict(data, feature, features[feature])
        pyfeats_features = get_pyfeats_features(rgb_img, mask)
        for feature in pyfeats_features:
            data = update_data_dict(data, feature, pyfeats_features[feature])

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

        # get prediction labels
        preds = model.predict(df_features[md_feat])
        print(preds)
        prediction_label = le.inverse_transform(preds)
        print(prediction_label)
        proba = model.predict_proba(df_features[md_feat]).max(axis=1).tolist()
        print(proba)

        # # prediction matching
        # if len(classes) == 2:
        #     if ((prediction_label == 'healthy') and (desease =='healthy')) or ((prediction_label == 'not healthy') and (desease !='healthy')):
        #         matching = True
        #     else:
        #         matching = False
        # else:
        #     matching = (prediction_label.lower() == f'{species}_{desease}'.lower())
                    
        # return {
        #     'class': prediction_label,
        #     'score': f'{100*float(proba):.2f}',
        #     'matching': 0, #matching,
        # }
     

    def dp_predict(img, model, class_names, species, desease):      
            img = cv.resize(np.array(img), tuple(model.layers[0].get_output_at(0).get_shape().as_list()[1:-1]))
            img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            
            # get prediction labels
            y = model.predict(img[tf.newaxis, ...])
            label_encoded = np.argmax(y, axis=-1)[0]
            prediction_label = class_names[label_encoded]
            prediction_accuracy = str(y.max())

            # prediction matching
            if len(class_names) == 2:
                if ((prediction_label == 'healthy') and (desease =='healthy')) or ((prediction_label == 'not_healthy') and (desease !='healthy')):
                    matching = True
                else:
                    matching = False
            else:
                matching = (prediction_label.lower() == f'{species}_{desease}'.lower())
                        
            return {
                'class': prediction_label,
                'score': f'{100*float(prediction_accuracy):.2f}',
                'matching': matching,
            }
   
    def multiprocess_worker(f, data_dir, comments, dp_model, class_names, ml_model, ml_model_dict):
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
        bgr_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        mask, _ = remove_bg(bgr_img)
        masked_img = PredictionController.generate_img_without_bg(mask, bgr_img)


        # Get DP prediction
        dp_predict_output = None
        if dp_model:
            dp_predict_output = PredictionController.dp_predict(rgb_img, dp_model, class_names, img_dict['img_species'], img_dict['img_desease'])
            if 'error' in dp_predict_output.keys():
                return create_response(data=dp_predict_output, status=500)

        # Get ML classification
        ml_predict_output = None
        # if ml_model:
        #     ml_predict_output = PredictionController.ml_predict(bgr_img, mask, masked_img, ml_model_dict, img_dict['img_species'], img_dict['img_desease'])

        # convert numpy array image to base64
        _, rgb_img = cv.imencode('.jpg', bgr_img)
        rgb_img = base64.b64encode(rgb_img).decode('utf-8')
        _, masked_img = cv.imencode('.jpg', masked_img)
        masked_img = base64.b64encode(masked_img).decode('utf-8')

        # add images to the dictionary
        img_dict['rgb_img'] = rgb_img
        img_dict['masked_img'] = masked_img
        
        if dp_predict_output:
            img_dict['dl_prediction'] = dp_predict_output
        if ml_predict_output:
            img_dict['ml_prediction'] = ml_predict_output

        del bgr_img, rgb_img, masked_img

        return img_dict 

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

        # Load ML model
        ml_model_dict = None
        if ml_model:
            # protect to LFI and RFI attacks
            ml_model = path.basename(ml_model)
            ml_model = ml_model.replace("%", '')
            if ml_model.find('/') != -1 or \
                ml_model.find('\\') != -1 or \
                ml_model.find('..') != -1 or \
                ml_model.find('.') != -1:
                return {'error': 'Incorrect model name don\'t try to hack us.'}
        
            model_path = f'../../data/models_saved/{ml_model}.joblib'
            if os.path.exists(model_path):
                df_features = pd.concat(list(tqdm(current_app._executor.map(PredictionController.get_ml_features,folders, repeat(data_dir)), total=len(folders))))
                current_app._executor.shutdown()
                print('Start loading ml model at :', dt.now())
                ml_model_dict = joblib.load(model_path)
                print('end features ml model at :', dt.now())
                PredictionController.ml_predict(ml_model_dict, df_features)
                print('end ml model at :', dt.now())
            else:
                return {'error': f'{ml_model} model not found'}   

        # Load ML model
        class_names = None
        models[dp_model] = None
        if dp_model:
            # protect to LFI and RFI attacks
            dp_model = path.basename(dp_model)
            dp_model = dp_model.replace("%", '')
            if dp_model.find('/') != -1 or \
                dp_model.find('\\') != -1 or \
                dp_model.find('..') != -1 or \
                dp_model.find('.') != -1:
                return {'error': 'Incorrect model name don\'t try to hack us.'}
        
            model_path = f'../../data/models_saved/{dp_model}.h5'
            if os.path.exists(model_path):
                custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m}
                models[dp_model] = load_model(model_path, custom_objects)
                f = h5py.File(model_path, mode='r')
                class_names_raw = None
                if 'class_names' in f.attrs and len(f.attrs['class_names']) > 0:
                    class_names_raw = f.attrs.get('class_names')
                    class_names = json.loads(class_names_raw)
                    f.close()
                else:
                    f.close()
                    return create_response(data={'error': 'Error don\'t have classes for classification.'}, status=500)
            else:
                return {'error': f'{dp_model} model not found'}   
        
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        # results = current_app._executor.map(PredictionController.multiprocess_worker, folders, data_dir), repeat(comments), repeat(models[dp_model]), repeat(class_names), repeat(ml_model), repeat(ml_model_dict))
        # current_model = models[dp_model]
        # current_model=None
        # results = current_app._pool.map(PredictionController.multiprocess_worker, (folders, data_dir, comments, current_model, class_names, ml_model, ml_model_dict))
        # results = current_app._pool.map(PredictionController.multiprocess_worker, [folders, data_dir, comments, current_model, class_names, ml_model, ml_model_dict])
        # results = current_app._pool.map(PredictionController.multiprocess_worker, [folders, repeat(data_dir), repeat(comments), repeat(current_model), repeat(class_names), repeat(ml_model), repeat(ml_model_dict)])
        # def proc(f):
        #     results_proc = []
        #     # for f in folders:
        #     results_proc.append(PredictionController.multiprocess_worker(f, data_dir, comments, models[dp_model], class_names, ml_model, ml_model_dict))
        #     return results_proc

        # pp = [current_app._pool.apply_async(proc(f)) for f in folders]
        # results = [p.get() for p in pp]
        # # p.start()
        # # results = p.get()
        # print(results)

        # results = parmap.starmap(PredictionController.multiprocess_worker, zip(folders), data_dir, comments, current_model, class_names, ml_model, ml_model_dict)
        print('start of multitreading : ', dt.now())

        # results = current_app._pool.starmap(PredictionController.multiprocess_worker, zip(folders, repeat(data_dir), repeat(comments), repeat(models[dp_model] if dp_model else None), repeat(class_names), repeat(ml_model), repeat(ml_model_dict)))
        # results = current_app._executor.map(PredictionController.multiprocess_worker,folders, repeat(data_dir), repeat(comments), repeat(models[dp_model] if dp_model else None), repeat(class_names), repeat(ml_model), repeat(ml_model_dict))

        output = []
        # for r in list(tqdm(current_app._executor.map(PredictionController.multiprocess_worker,folders, repeat(data_dir), repeat(comments), repeat(models[dp_model] if dp_model else None), repeat(class_names), repeat(ml_model), repeat(ml_model_dict)), total=len(folders))):
        #     output.append(r)

        print('end of mulßtitreading : ', dt.now(), 'lenght of results : ', len(output))
        # current_app._executor.shutdown()

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

        # Load ML model
        ml_model_dict = None
        if ml_model:
            # protect to LFI and RFI attacks
            ml_model = path.basename(ml_model)
            ml_model = ml_model.replace("%", '')
            if ml_model.find('/') != -1 or \
                ml_model.find('\\') != -1 or \
                ml_model.find('..') != -1 or \
                ml_model.find('.') != -1:
                return {'error': 'Incorrect model name don\'t try to hack us.'}
        
            model_path = f'../../data/models_saved/{ml_model}.joblib'
            if os.path.exists(model_path):
                print('strat loading ml model at :', dt.now())
                ml_model_dict = joblib.load(model_path)
                print('end loading ml model at :', dt.now())
            else:
                return {'error': f'{ml_model} model not found'}   

        # Load ML model
        class_names = None
        models[dp_model] = None
        if dp_model:
            # protect to LFI and RFI attacks
            dp_model = path.basename(dp_model)
            dp_model = dp_model.replace("%", '')
            if dp_model.find('/') != -1 or \
                dp_model.find('\\') != -1 or \
                dp_model.find('..') != -1 or \
                dp_model.find('.') != -1:
                return {'error': 'Incorrect model name don\'t try to hack us.'}
        
            model_path = f'../../data/models_saved/{dp_model}.h5'
            if os.path.exists(model_path):
                print('strat loading dp model at :', dt.now())
                custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m}
                models[dp_model] = load_model(model_path, custom_objects)
                f = h5py.File(model_path, mode='r')
                class_names_raw = None
                if 'class_names' in f.attrs and len(f.attrs['class_names']) > 0:
                    class_names_raw = f.attrs.get('class_names')
                    class_names = json.loads(class_names_raw)
                    f.close()
                else:
                    f.close()
                    return create_response(data={'error': 'Error don\'t have classes for classification.'}, status=500)
                print('end loading dp model at :', dt.now())
            else:
                return {'error': f'{dp_model} model not found'}   
 
        # add comment if it exits
        img_dict['comment'] = ''
        if len(comments)>0 and class_name:
            cmt = [x for x in comments if (x['species'] == img_dict['img_species']) and (x['desease'] == img_dict['img_desease']) and (x['img_num'] == img_dict['img_num'])]
            if len(cmt)>0:
                img_dict['comment'] = cmt[0]['comment']
        
        # Image processing
        bgr_img = np.array(Image.open(io.BytesIO(base64.b64decode(b64File))))
        bgr_img = cv.resize(np.array(bgr_img), (256,256))
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        mask, _ = remove_bg(bgr_img)

        # Get DP prediction
        dp_predict_output = None
        if dp_model:
            dp_predict_output = PredictionController.dp_predict(bgr_img, models[dp_model], class_names, img_dict['img_species'], img_dict['img_desease'])
            if 'error' in dp_predict_output.keys():
                return create_response(data=dp_predict_output, status=500)

        # Get ML classification
        ml_predict_output = None
        if ml_model:
            masked_img = PredictionController.generate_img_without_bg(mask, bgr_img)
            ml_predict_output = PredictionController.ml_predict(bgr_img, mask, masked_img, ml_model_dict, img_dict['img_species'], img_dict['img_desease'])

        # convert numpy array image to base64
        _, rgb_img = cv.imencode('.jpg', bgr_img)
        rgb_img = base64.b64encode(rgb_img).decode('utf-8')
        _, masked_img = cv.imencode('.jpg', masked_img)
        masked_img = base64.b64encode(masked_img).decode('utf-8')

        # add images to the dictionary
        img_dict['rgb_img'] = rgb_img
        img_dict['masked_img'] = masked_img
        
        if dp_predict_output:
            img_dict['dl_prediction'] = dp_predict_output
        if ml_predict_output:
            img_dict['ml_prediction'] = ml_predict_output

        del bgr_img, rgb_img, masked_img

        return create_response(data={str(k): v for k, v in enumerate([img_dict])})

    def predict(b64img, model_name, should_remove_bg):
        
        # protect to LFI and RFI attacks
        model_name = path.basename(model_name)
        model_name = model_name.replace("%", '')
        if model_name.find('/') != -1 or \
           model_name.find('\\') != -1 or \
           model_name.find('..') != -1 or \
           model_name.find('.') != -1:
           return create_response(data={'error': 'Incorrect model name don\'t try to hack us.'}, status=500)
        
        model_path = f'../../data/models_saved/{model_name}.h5'
        model_exist = os.path.exists(model_path)
        
        if model_exist:
            custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m}
            models[model_name] = load_model(model_path, custom_objects)
            f = h5py.File(model_path, mode='r')
            class_names_raw = None
            if 'class_names' in f.attrs and len(f.attrs['class_names']) > 0:
                class_names_raw = f.attrs.get('class_names')
                class_names = json.loads(class_names_raw)
                f.close()
            else:
                f.close()
                return create_response(data={'error': 'Error don\'t have classes for classification.'}, status=500)

            model = models[model_name]
            base64_decoded = base64.b64decode(b64img)
            image = Image.open(io.BytesIO(base64_decoded))
            image_np = np.array(image)
            _, im_withoutbg_b64 = remove_bg(image_np)
            
            # call remove background function
            if should_remove_bg:
                _, new_img = im_withoutbg_b64
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



            