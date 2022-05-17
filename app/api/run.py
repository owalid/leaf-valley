from flask import jsonify
import os
from flask import Flask
from flask_cors import CORS

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

    CORS(app, origins=r"*")  # add CORS

    # import and register blueprints
    from routes import predict_bp

    # why blueprints http://flask.pocoo.org/docs/1.0/blueprints/
    app.register_blueprint(analysis_bp.mod)
    app.register_blueprint(predict_bp.mod)

    return app


# sets up the app
app = create_app()

if __name__ == '__main__':
  app.run(host='127.0.0.1', debug=True)