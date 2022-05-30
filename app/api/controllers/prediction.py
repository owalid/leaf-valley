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

from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
current_dir = current_dir[:current_dir.rfind(path.sep)]
current_dir = current_dir[:current_dir.rfind(path.sep)]
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from utilities.remove_background_functions import remove_bg

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
            models[model_name] = keras.models.load_model(model_path)

            model = models[model_name]
            print(type(model))
            base64_decoded = base64.b64decode(b64img)
            image = Image.open(io.BytesIO(base64_decoded))
            image_np = np.array(image)
            print(image_np[0])
            im_withoutbg_b64 = ''
            
            
            # call remove background function
            _, new_img = remove_bg(image_np)
            
             # convert numpy array image to base64
            _, img_arr = cv.imencode('.jpg', new_img)
            im_withoutbg_b64 = base64.b64encode(img_arr).decode('utf-8')
            
            # call predict function
            '''
                y = model.predict(im_arr) # TODO uncomment this when remove_bg function is ready and model.predict(im_arr) is ready
            '''
            
            prediction = {
                'prediction': 'healthy',
                'accuracy': 0.9,
                'im_withoutbg_b64': im_withoutbg_b64
            }
            return create_response(data={'result': prediction})
        else:
            return create_response(data={'error': f'{model_name} model not found'}, status=500)
            
        
    def get_leaves_by_dir(directory):
        result = []
        return create_response(data={'result': result})
