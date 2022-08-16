'''
  CLI used to manage ML classification.
'''
import os
import sys
import random
import joblib

import pandas as pd
import numpy as np

import argparse as ap
from inspect import getsourcefile

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

def set_models_dict(classType, classModels, verbose):
  xgc = xgb.XGBClassifier(use_label_encoder=False, objective='binary:logistic' if classType == 'HEALTHY' else 'multi:softmax', eval_metric=f1_score, n_estimators=500, n_jobs=-1, verbose=verbose)
  rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, verbose=verbose)
  etc = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, verbose=verbose)
  _dict = dict(('xgc', xgc()), ('rfc', rfc()), ('etc', etc()))
  return { md: _dict[md.lower()] for md in classModels}

def split_data(df):
    features = [f for f in df.columns if f not in ['classes','split']]
    X_train = df.loc[df.split=='train'][features]
    y_train = df.loc[df.split=='train']['classes']
    X_test  = df.loc[df.split=='test'][features]
    y_test  = df.loc[df.split=='test']['classes']
    return X_train, y_train, X_test, y_test

# Datat normalization
def data_normalization(df, scaler, save=False, fit=False):
  # fit data normalization
  if fit:
    scaler.fit(df)

  # transform data normalization
  return scaler, scaler.transform(df).astype(np.float32)

# Label encoding
def label_encoding(df, le, class_type, save=False, fit=False):
  # Class agregation
  if class_type == 'HEALTHY':
    df_features['classes'] = df_features.classes.apply(lambda c: 'healthy' if 'healthy' in c.lower() else 'not_healthy')
  if class_type == 'PLANTS':
    df_features['classes'] = df_features.classes.apply(lambda c: c.split('___')[0])

  # fit data normalization
  if fit:
    le.fit(df)

  # transform data normalization
  return le, le.transform(df)

# Save model
def save_model_func(model, scaler, le, features,class_type, md_label, dst_path):
    path = os.path.join(dst_path, f'ML_{md_label.upper()}_{class_type}.pkl.z')
    if os.path.exists(path):
        os.remove(path)

    md_dict = {}
    md_dict['ml_model'] = model
    md_dict['ml_scaler'] = scaler
    md_dict['ml_label_encoder'] = le
    md_dict['ml_features'] = features
    joblib.dump(md_dict, path)
    local_print(f'Info : model {md_label} has been saved !!!§')

# fit models
def fit_models(X_train, y_orig, classification_models, classification_types, save, options_dataset, verbose):
  # Data normalization
  scaler, X_train = data_normalization(X_train, scaler_dict[normalize_type], save=save, fit=True)

  # Processing
  models_dict = {}
  for class_type in classification_types:
    # Label encoding
    le, y_train = label_encoding(y_orig, LabelEncoder(), fit=True)

    # Set models dictionary
    models_dict[class_type] = set_models_dict(class_type, classification_models, verbose)

    # Fit models
    for md in models_dict[class_type].keys():
        models_dict[class_type][md].fit(X_train, y_train)
        if save:
          save_model_func(models_dict[class_type][md], scaler, le, X_train.columns, class_type, md, options_dataset, 'data/models', verbose)

# predict models
def predict_models(X_test, y_orig, classification_models, verbose):
  # Data normalization
  scaler, X_train = data_normalization(X_test, scaler_dict[normalize_type])

  # Processing
  models_dict = {}
  for class_type in classification_types:
    # Label encoding
    le, y_train = label_encoding(y_orig, LabelEncoder(), fit=True)

    # Set models dictionary
    models_dict[class_type] = set_models_dict(class_type, classification_models, verbose)

    # Fit models
    for md in models_dict[class_type].keys():
        models_dict[class_type][md].fit(X_train, y_train)
        if save:
          save_model_func(models_dict[class_type][md], scaler, le, X_train.columns, class_type, md, options_dataset, 'data/models', verbose)


# MAIN
if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument("-src", "--src-directory", required=False, type=str, default='data/preprocess',
                        help='Directory source who can find data for the classification. default: data/preprocess/ml_classification')
    parser.add_argument("-f", "--filename", required=True, type=str, 
                        help='Basename of the file input data without extension')
    parser.add_argument("-dst", "--preprocessed-data", required=False, type=str, default='data/preprocess/ml_classification',
                        help='Path to save or to get the preprocessed data. default: data/preprocess/ml_classification')
    parser.add_argument("-sd", "--save-data", required=False,
                        action='store_true', default=False, help='Save options_datasets json file and converted data from h5 format to DataFrame one with flag train/test flag')
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
    file_basename         = args.filename
    preprocessed_data           = args.preprocessed_data
    classification_step   = args.classification_step
    classification_types  = ['PLANTS', 'HEALTHY', 'PLANTS_DESEASES'] if args.classification_type.upper() == 'ALL' else args.classification_type.split(',')
    classification_models = ['XGB', 'ETC', 'RFC'] if args.classification_models.upper() == 'ALL' else args.classification_models.split(',')
    normalize_type        = args.normalize_type
    save_data             = args.save_data
    save_model            = args.save_model
    save_plot             = args.save_plot
    save_scores           = args.save_scores
    verbose               = args.verbose

    if not os.path.exists(preprocessed_data):
      os.makedirs(preprocessed_data)
      local_print(f'\nInfo : folder {preprocessed_data} was created successfully !!\n', verbose)

    if classification_step in ['ALL', 'LOAD_DATA'] :
      if not os.path.exists(os.path.join(src_directory, f'{file_basename}.h5')):
        print(f'Error : filename ({file_basename}.h5) not a h5 file or doesn\'t exist')
        exit(1)
      else:
        df_features, options_dataset = load_data_from_h5(src_directory, f'{file_basename}.h5', verbose)
        
        if save_data:
          df_features.to_pickle(os.path.join(preprocessed_data, file_basename+".pkl"))
          joblib.dump(options_dataset, os.path.join(preprocessed_data, file_basename+"_options_dataset.joblib"))
        
        if classification_step == 'LOAD_DATA':
          local_print('Info : end of the load data job')
          exit(0)

    # load data if not yet loaded
    if classification_step in ['FIT_MODEL', 'PREDICT_MODEL', 'FIT_PREDICT_MODEL'] :
        if not os.path.exists(os.path.join(preprocessed_data, file_basename+".pkl")):
            local_print(f'Error : the file {file_basename}.pkl does not exist')
            exit(1)
        df_features = pd.read_pickle(os.path.join(preprocessed_data, file_basename+".pkl"))
        options_dataset =  joblib.load(os.path.join(preprocessed_data, file_basename+"_options_dataset.joblib"))

    # Split data
    X_train, y_train, X_test, y_test = split_data(df_features)

    if classification_step != 'PREDICT_MODEL':
      # fit models
      fit_models(X_train, y_train, classification_models, classification_types, save_model, options_dataset, verbose)
      if classification_step == 'FIT_MODEL':
          local_print('Info : end of the models fit', verbose)
          exit(0)








    random.seed(42)