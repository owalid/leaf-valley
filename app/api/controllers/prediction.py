from itertools import chain
from utils.mixins import create_response, serialize_list

class PredictionController:
    def get_models():
        models = []
        return create_response(data={'models': models})

    def get_leaves_by_dir(directory):
        result = []
        return create_response(data={'result': result})