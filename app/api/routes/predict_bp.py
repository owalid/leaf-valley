from statistics import mode
from flask import Blueprint, jsonify
from flask import request
from utils.mixins import create_response, serialize_list
from controllers.prediction import PredictionController
import os
mod = Blueprint('predict_routes', __name__, url_prefix='/api/models')

@mod.route('/')
def get_models():
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
