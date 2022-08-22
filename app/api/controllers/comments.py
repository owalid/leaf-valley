import os
import re
import json

import sys
from inspect import getsourcefile

from utils.mixins import create_response

FLASK_ENV = os.environ.get("FLASK_ENV", "dev")

if FLASK_ENV != 'prod':
    current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
    current_dir = current_dir[:current_dir.rfind(os.path.sep)]
    current_dir = current_dir[:current_dir.rfind(os.path.sep)]
    sys.path.insert(0, current_dir[:current_dir.rfind(os.path.sep)])

from utilities.utils import safe_open_w

class CommentsController:
    FLASK_ENV = os.environ.get("FLASK_ENV", "dev")
    comment_filename = 'data/plants_comments.json' if FLASK_ENV == 'prod' else '../data/plants_comments.json'
    
    def process_comment(method, comment):
        comment = re.sub('[^A-Za-z0-9]+', '', comment)
        with safe_open_w(CommentsController.comment_filename, option_open="r") as json_file:
            try:
                comments = json.load(json_file)
            except:
                comments = []

        # insert comment
        if method == 'insert':
            comments.append(comment)
        if method == 'update':
            comments = [x for x in comments if not ((x['species'] == comment['species']) and (x['desease'] == comment['desease']) and (x['img_num'] == comment['img_num']))]
            comments.append(comment)
        if method == 'delete':
            comments = [x for x in comments if not ((x['species'] == comment['species']) and (x['desease'] == comment['desease']) and (x['img_num'] == comment['img_num']))]

        # Save comments file
        with safe_open_w(CommentsController.comment_filename, option_open="w") as json_file:
            json.dump(comments, json_file, indent=2)

        return create_response(data={'result': f'{method} with success'})