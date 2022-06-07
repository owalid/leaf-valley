from flask import Blueprint, jsonify
from utils.mixins import create_response, serialize_list
from controllers.prediction import PredictionController

mod = Blueprint('predict_routes', __name__, url_prefix='/api/models')

@mod.route('/')
def get_models():
    return PredictionController.get_models()

@mod.route('/predict', methods=[ 'POST' ])
def get_prediction():
    return PredictionController.predict()
