import os
from flask import Blueprint, request
from controllers.prediction import PredictionController
from utils.mixins import create_response

mod = Blueprint('predict_routes', __name__, url_prefix='/api/models')

@mod.route('/')
def get_models():
    print("hello world /")
    return PredictionController.get_models()

@mod.route('/predict', methods=[ 'POST' ])
def get_prediction():
    data = request.get_json()
    if data:
        img = data.get('img')
        model_name = data.get('model_name')
        should_remove_bg = data.get('should_remove_bg')

        if img and model_name:
            return PredictionController.predict(img, model_name, should_remove_bg)
        return create_response(data={'error': 'Incorrect body'}, status=500)
    else:
        return create_response(data={'error': 'Incorrect body'}, status=500)

@mod.route('/plants')
def get_plants():
    return PredictionController.get_plants()

@mod.route('/classes')
def get_classes():
    return PredictionController.get_classes()

@mod.route('/random-img', methods=[ 'POST' ])
def get_randomimage():
    data = request.get_json()
    if data:
        nb_img = data.get('number_img')
        spacies = data.get('spacies')
        desease = data.get('desease')
        ml_model = data.get('ml_model')
        dl_model = data.get('dl_model')
        
    return PredictionController.get_randomimage(nb_img, spacies, desease, ml_model, dl_model)


@mod.route('/select-img', methods=[ 'POST' ])
def get_selectedimge():
    data = request.get_json()
    if data:
        b64File    = data.get('b64Files')
        ml_model   = data.get('ml_model')
        dl_model   = data.get('dl_model')
        class_name = data.get('class_name')
        
    return PredictionController.get_selectedimage(class_name, b64File, ml_model, dl_model)
