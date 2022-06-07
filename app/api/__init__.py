import os

from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy


def create_app(test_config=None):
    """
    The flask application factory. To run the app somewhere else you can:
    ```
    from api import create_app
    app = create_app()

    if __main__ == "__name__":
        app.run()
    """
    app = Flask(__name__)

    CORS(app)  # add CORS

    # check environment variables to see which config to load
    # instantiate database object

    # import and register blueprints
    # from routes import cow_bp

    # why blueprints http://flask.pocoo.org/docs/1.0/blueprints/
    # app.register_blueprint(main.main)

    # register error Handler
    # app.register_error_handler(Exception, all_exception_handler)

    return app
