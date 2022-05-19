from itertools import chain
import os
import base64
from PIL import Image
from itsdangerous import base64_encode
import numpy as np
import io
from utils.mixins import create_response, serialize_list
from tensorflow import keras

global models

class PredictionController:
    def get_models():
        models_dir_path = '../../data/models_saved'
        models_dir_exist = os.path.isdir(models_dir_path)
        all_models = []
        if models_dir_exist:
            all_models = [f.split(".")[-2] for f in os.listdir(models_dir_path) if '__' and f.split(".")[-1] == 'h5' in f]
        return create_response(data={'models': all_models})
    

    def predict(b64img, model_name):
        model_path = f'../../data/models_saved/{model_name}.h5'
        model_exist = os.path.exists(model_path)
        
        if model_exist:
            models[model_name] = keras.models.load_model(model_path)

            model = models[model_name]
            base64_decoded = base64.b64decode(b64img)
            image = Image.open(io.BytesIO(base64_decoded))
            image_np = np.array(image)
            im_withoutbg_b64 = ''
            
            # call remove background function
            '''
                im_arr = remove_bg(img_np) # TODO uncomment this when remove_bg function is ready
                im_bytes = im_arr.tobytes()
                im_withoutbg_b64 = base64.b64encode(im_bytes)
            '''
            
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