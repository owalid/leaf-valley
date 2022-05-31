from itertools import chain
import cv2 as cv
import os
import base64
from PIL import Image
import numpy as np
import io
from utils.mixins import create_response, serialize_list
from tensorflow import keras
import sys
import json
import h5py
import tensorflow as tf
from keras.models import load_model

from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
current_dir = current_dir[:current_dir.rfind(path.sep)]
current_dir = current_dir[:current_dir.rfind(path.sep)]
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from utilities.remove_background_functions import remove_bg
# todo decoment this when deep learning classifier is merged: from process.deep_learning.metrics import recall_m, precision_m, f1_m

global models
models = {}

class PredictionController:
    def get_models():
        models_dir_path = '../../data/models_saved'
        models_dir_exist = os.path.isdir(models_dir_path)
        all_models = []
        if models_dir_exist:
            all_models = [f.split(".")[-2] for f in os.listdir(models_dir_path) if '__' and f.split(".")[-1] == 'h5' in f]
        return create_response(data={'models': all_models})
    
   
    
    def predict(b64img, model_name):
        
        # TODO REMOVE THIS WHEN PR OF DEEP-LEARNING IS MERGED
        import tensorflow as tf
        import numpy as np
        from keras import backend as K

        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))
        
        # END TODO
        
        print(model_name)
        
        # protect to lfi and RFI
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
            im_withoutbg_b64 = ''
            
            # call remove background function
            _, new_img = remove_bg(image_np)
            new_img = cv.resize(new_img, tuple(model.layers[0].get_output_at(0).get_shape().as_list()[1:-1]))
            
            # Get prediction labels
            y = model.predict(new_img[tf.newaxis, ...])
            label_encoded = np.argmax(y, axis=-1)[0]
            prediction_label = class_names[label_encoded]
            prediction_accuracy = str(y.max())
            
             # convert numpy array image to base64
            _, img_arr = cv.imencode('.jpg', new_img)
            im_withoutbg_b64 = base64.b64encode(img_arr).decode('utf-8')
            prediction_data = {
                'prediction': prediction_label,
                'accuracy': str(prediction_accuracy),
                'im_withoutbg_b64': im_withoutbg_b64
            }
            return create_response(data=prediction_data)
        else:
            return create_response(data={'error': f'{model_name} model not found'}, status=500)
            
        
    def get_leaves_by_dir(directory):
        result = []
        return create_response(data={'result': result})
