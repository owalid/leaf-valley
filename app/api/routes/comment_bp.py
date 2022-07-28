from flask import Blueprint, request
from controllers.comments import CommentsController


mod = Blueprint('comment_routes', __name__, url_prefix='/api/comment')

@mod.route('/', methods=[ 'POST' ])
def process_comment():
    data = request.get_json()
    if data:
        method = data.get('method')
        comment = data.get('comment')
    
    return CommentsController.process_comment(method, comment)