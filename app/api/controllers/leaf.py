from itertools import chain
from utils.mixins import create_response, serialize_list

class LeafController:
    def get_all_leaves():
        leaves = []
        return create_response(data={'leaves': leaves})

    def get_leaves_by_dir(directory):
        leaves = []
        return create_response(data={'leaves': leaves})

    def get_leave_by_disease(disease):
        leaves = []
        return create_response(data={'leaves': leaves})