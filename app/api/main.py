import concurrent.futures
from dotenv import load_dotenv, find_dotenv
import os
from flask import Flask
from flask_cors import CORS
from flask import g

load_dotenv(find_dotenv())
FLASK_ENV = os.environ.get("FLASK_ENV", "dev")

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
    from routes import comment_bp, predict_bp

    # why blueprints http://flask.pocoo.org/docs/1.0/blueprints/
    app.register_blueprint(predict_bp.mod)
    app.register_blueprint(comment_bp.mod)

    return app


# sets up the app
app = create_app()

if __name__ == '__main__':
    try:
        FLASK_ENV = os.environ.get("FLASK_ENV", "dev")
        print("FLASK_ENV:", FLASK_ENV)
        port = 5000 if FLASK_ENV != "prod" else 80
        debug = FLASK_ENV != "prod"
        
        # import s3module
        from modules.s3_module import S3Module
        print("[main] init s3 end")
        s3_module = S3Module()
        print("[main] end init s3 end")
        
        app._executor = concurrent.futures.ProcessPoolExecutor(max_workers=((1+os.cpu_count()//5)*5))
        app.run(host='0.0.0.0', debug=debug, port=port)
    except KeyboardInterrupt:
        if app._executor:
            app._executor.shutdown()
