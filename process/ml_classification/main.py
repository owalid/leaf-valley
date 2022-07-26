'''
  CLI used to manage ML classification.
'''
import os
import sys
import random
import argparse as ap
from inspect import getsourcefile

import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, os.path.sep.join(current_dir.split(os.path.sep)[:-2]))

from utilities.utils import local_print
import load_data_from_h5

scaler_dict = dict(('NORM_MINMAX', MinMaxScaler()), ('NORM_STANDARSCALER', StandardScaler()))

def set_models(classType, classModels, verbose):
  xgc = xgb.XGBClassifier(use_label_encoder=False, objective='binary:logistic' if classType == 'HEALTHY' else 'multi:softmax', eval_metric=f1_score, n_estimators=500, n_jobs=-1, verbose=verbose)
  rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, verbose=verbose)
  etc = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, verbose=verbose)
  _dict = dict(('xgc', xgc()), ('rfc', rfc()), ('etc', etc()))
  return { md: _dict[md.lower()] for md in classModels}

# MAIN
if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument("-src", "--src-directory", required=False, type=str, default='data/preprocess',
                        help='Directory source who can find data for the classification. default: data/process/ml_classification')
    parser.add_argument("-f", "--filename", required=True, type=str, 
                        help='Filename input data format h5 for raw data or pkl for train/test data')
    parser.add_argument("-dst", "--destination", required=False, type=str, default='data/process/ml_classification',
                        help='Path to save the data. default: data/process/ml_classification')
    parser.add_argument("-sd", "--save-data", required=False,
                        action='store_true', default=False, help='Save train/test data and options_datasets')
    parser.add_argument("-nortype", "--normalize-type", required=False, type=str, default='NORM_MINMAX',
                        help='Normalize data (NORM_STANDARSCALER or NORM_MINMAX normalization) (Default: NORM_MINMAX)')
    parser.add_argument("-cs", "--classification-step", required=False, type=str, default="ALL",
                        help='Classification step: LOAD_DATA, FIT_MODEL, PREDICT_MODEL, FIT_PREDICT_MODEL, ALL (default)')
    parser.add_argument("-cm", "--classification-models", required=False, type=str, default="ALL",
                        help='Classification models: XGB, ETC, RFC, ALL (default). Example -cm=RFC,ETC')
    parser.add_argument("-ct", "--classification-type", required=False, type=str, default="ALL",
                        help='Classification type: PLANTS, HEALTHY, PLANTS_DESEASES classes, ALL (default)')
    parser.add_argument("-sm", "--save-model", required=False,
                        action='store_true', default=True, help='Save model')
    parser.add_argument("-sp", "--save-plot", required=False,
                        action='store_true', default=True, help='Save heatmap plot')
    parser.add_argument("-ss", "--save-scores", required=False,
                        action='store_true', default=True, help='Save scores')
    parser.add_argument("-v", "--verbose", required=False,
                        action='store_true', default=False, help='Verbose')

    args = parser.parse_args()
    print(args)

    src_directory         = args.src_directory
    filename              = args.filename
    destination           = args.destination
    classification_step   = args.classification_step
    classification_type   = args.classification_type
    classification_models = ['XGB', 'ETC', 'RFC'] if args.classification_models.upper() == 'ALL' else args.classification_models.split(',')
    normalize_type        = args.normalize_type
    save_data             = args.save_data
    save_model            = args.save_model
    save_plot             = args.save_plot
    save_scores           = args.save_scores
    verbose               = args.verbose

    file_basename = filename.split('.')[0]

    if not os.path.exists(destination):
      os.makedirs(destination)
      local_print(f'\nInfo : folder {destination} was created successfully !!\n', verbose)

    if classification_step in ['ALL', 'LOAD_DATA'] :
      if not (filename.split('.')[-1] == 'h5' and os.path.exists(os.path.join(src_directory, filename))): 
        print(f'Error : filename ({filename}) not a h5 file or doesn\'t exist')
        exit(1)
      else:
        df_features, options_dataset = load_data_from_h5(src_directory, filename, verbose)
        # SPlit manually data into train/test to save it
        df_features = shuffle(df_features)

        df_features.loc[random.sample(df_features.index.to_list(), int(.7*len(df_features))),'split'] = 'train'
        df_features['split'].fillna('test', inplace=True)
        
        if save_data:
          df_features.to_pickle(os.path.join(destination, file_basename+".pkl"))
        
        if classification_step == 'LOAD_DATA':
          local_print('Info : end of the load data job')
          exit(0)

    if not os.path.exists(os.path.join(destination, file_basename+".pkl")):
        local_print(f'Error : the file {file_basename}.pkl does not exist')
        exit(1)
    
    else:
      # load data if not yet loaded
      if classification_step in ['FIT_MODEL', 'PREDICT_MODEL', 'FIT_PREDICT_MODEL'] :
        df_features = pd.read_pickle(os.path.join(destination, file_basename+".pkl"))

      # data normalization
      features = [f for f in df_features.columns if f not in ['classes','split']]
      scaler = scaler_dict[normalize_type] 
      scaler.fit(df_features[features])
      df_features[features]=scaler.transform(df_features[features]).astype(np.float32)

      # Class agregation
      if classification_type == 'HEALTHY':
        df_features['classes'] = df_features.classes.apply(lambda c: 'healthy' if 'healthy' in c.lower() else 'not_healthy')
      if classification_type == 'PLANTS':
        df_features['classes'] = df_features.classes.apply(lambda c: c.split('___')[0])
      
      # Label Encoding
      le = LabelEncoder()

      # Coding of the label for plants with possible diseases
      le.fit(df_features.classes)
      df_features['label'] = le.transform(df_features.classes)

      # Set models
      models_dict = set_models(classification_type, classification_models, verbose)









    random.seed(42)

