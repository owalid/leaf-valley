from flask import Blueprint, jsonify
from utils.mixins import create_response, serialize_list
from controllers.leaf import LeafController

mod = Blueprint('analysis_routes', __name__, url_prefix='/api/leaf')

@mod.route('/')
def get_all_leaves():
    return LeafController.get_all_leaves()

@mod.route('/<directory>')
def get_leaves_by_dir(directory):
    return LeafController.get_leaves_by_dir(directory)

@mod.route('/<disease>')
def get_leave_by_disease(disease):
    return LeafController.get_leave_by_disease(disease)
