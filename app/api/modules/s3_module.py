import os
import io
from PIL import Image
import pandas as pd
import numpy as np
import boto3


class S3Module():
    
    __instance = None
    
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(S3Module,cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance
    
    def __init__(self):
        if (self.__initialized): return
        
        self.__initialized = True
        FLASK_ENV = os.environ.get("FLASK_ENV", "dev")
        self.S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID", None)
        self.S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY", None)
        self.S3_BASE_ENDPOINT_URL = os.environ.get("S3_BASE_ENDPOINT_URL", None)
        self.S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", None)
        self.S3_MODELS_FOLDER = os.environ.get("S3_MODELS_FOLDER", None)
        self.S3_DATASET_FOLDER = os.environ.get("S3_DATASET_FOLDER", None)
        
        if FLASK_ENV == "dev" or self.S3_ACCESS_KEY_ID is None or self.S3_SECRET_ACCESS_KEY is None or self.S3_BASE_ENDPOINT_URL is None or self.S3_BUCKET_NAME is None:
            print("S3module => dev mode or missing environment variables")
            return
        
        self.session = boto3.Session(aws_access_key_id=self.S3_ACCESS_KEY_ID, aws_secret_access_key=self.S3_SECRET_ACCESS_KEY, region_name='fr-par')
        self.s3_client = self.get_s3_client()
        self.s3_resource = self.get_s3_resource()
        self.models_list = self.get_models_list()
        self.files_leafs = self.get_files_leafs()
    
    def get_s3_client(self):
        return self.session.client('s3', endpoint_url=self.S3_BASE_ENDPOINT_URL)
    
    def get_s3_resource(self):
        # session = boto3.Session(aws_access_key_id=self.S3_ACCESS_KEY_ID, aws_secret_access_key=self.S3_SECRET_ACCESS_KEY)
        return self.session.resource('s3', endpoint_url=self.S3_BASE_ENDPOINT_URL)
    
    def get_models_list(self):
        model_bucket = self.s3_resource.Bucket(self.S3_BUCKET_NAME)
        models_list = []
        
        for model_bucket_object in model_bucket.objects.filter(Prefix=self.S3_MODELS_FOLDER):
            models_list.append(model_bucket_object.key.split("/")[-1])
            
        return models_list

    def get_files_leafs(self):
        model_bucket = self.s3_resource.Bucket(self.S3_BUCKET_NAME)
        files_leafs = []
        
        for model_bucket_object in model_bucket.objects.filter(Prefix=self.S3_DATASET_FOLDER):
            files_leafs.append(model_bucket_object.key)
            
        return files_leafs
    
    def get_df_leafs(self):
        '''
            Get dataframe from path of datasets
            path: path of datasets

            return pandas dataframe
        '''
        
        all_folder = self.get_folders_leafs()
        df = pd.DataFrame(columns=['number_img', 'disease',
                        'disease_family', 'healthy', 'specie'], index=all_folder)

        for name_folder in all_folder:
            
            files = [f for f in self.files_leafs if name_folder in f]
            name_splited = name_folder.split('___')
            df.loc[name_folder].specie = name_splited[0].lower()
            df.loc[name_folder].number_img = len(files)
            df.loc[name_folder].disease = name_splited[-1].lower()
            df.loc[name_folder].disease_family = df.loc[name_folder].disease.split(
                '_')[-1].replace(')', '')
            df.loc[name_folder].healthy = name_splited[-1] == 'healthy'
            
        return df
    
    def get_folders_leafs(self):
        all_folder = list(set([f.split('/')[1] for f in self.files_leafs]))
        return all_folder
    
    def get_image_from_path(self, path):
        '''
            Get image from path
            path: path of image

            return image
        '''
        return np.array(Image.open(io.BytesIO(self.s3_resource.Object(self.S3_BUCKET_NAME, path).get()['Body'].read())))