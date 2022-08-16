'''
  CLI used to manage ML classification.
'''
from ctypes import sizeof
import os
import sys
import random
import joblib

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import argparse as ap
from inspect import getsourcefile

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, os.path.sep.join(current_dir.split(os.path.sep)[:-2]))

from utilities.utils import local_print
from load_data_from_h5 import load_data_from_h5

scaler_dict = {
                'NORM_MINMAX': MinMaxScaler(), 
                'NORM_STANDARSCALER': StandardScaler()
                }

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
  df = scaler.transform(df).astype(np.float32)

  return scaler, df

# Label encoding
def label_encoding(df, le, class_type, fit=False):
  # Class agregation
  if class_type == 'HEALTHY':
    df['classes'] = df.classes.apply(lambda c: 'healthy' if 'healthy' in c.lower() else 'not_healthy')
  if class_type == 'PLANTS':
    df['classes'] = df.classes.apply(lambda c: c.split('___')[0])

  # fit label encoding  
  if fit:
    le.fit(df['classes'])

  # transform label encoding
  df['classes'] = le.transform(df['classes'])

  return le, df

# Save model
def save_model_func(md_label, model, scaler, le, features, class_type, options_dataset, dst_path, verbose):
    path = os.path.join(dst_path, f'ML_{md_label.upper()}_{class_type}.pkl.z')
    if os.path.exists(path):
        os.remove(path)

    md_dict = {}
    md_dict['ml_model'] = model
    md_dict['ml_scaler'] = scaler
    md_dict['ml_label_encoder'] = le
    md_dict['ml_features'] = features
    md_dict['options_dataset'] = options_dataset
    joblib.dump(md_dict, path)
    local_print(f'Info : model {md_label} has been saved !!!§', verbose)

# Load model
def load_model_func(md_label, class_type, dst_path, verbose):
    path = os.path.join(dst_path, f'ML_{md_label.upper()}_{class_type}.pkl.z')
    if not os.path.exists(path):
        print(f"Error : the ML_{md_label.upper()}_{class_type}.pkl.z does not exist !!!")
        exit()

    md_dict = joblib.load(path)
    model = md_dict['ml_model']
    scaler = md_dict['ml_scaler']
    le = md_dict['ml_label_encoder']
    features = md_dict['ml_features']
    options_dataset = md_dict['options_dataset']
    local_print(f'Info : model {md_label} has been load !!!§', verbose)

    return model, scaler, le, features, options_dataset

# fit models
def fit_models(X_train, y_orig, classification_models, classification_types, options_dataset, save ,md_dst, verbose):
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
    for md_label in models_dict[class_type].keys():
        models_dict[class_type][md_label].fit(X_train, y_train)
        if save:
          save_model_func( md_label, models_dict[class_type][md_label], scaler, le, X_train.columns, class_type,options_dataset, md_dst, verbose)

# Accuracy Classification Report
def accuracy_classification_report(y_test, y_pred, classes, msg = '', filename=''):
  confusion_mtx = {
                    'y_Actual': np.array(y_test),
                    'y_Predicted': y_pred
                }

  confusion_df = pd.DataFrame(confusion_mtx, columns=['y_Actual','y_Predicted'])    
  score = (stats.spearmanr(confusion_df['y_Actual'], confusion_df['y_Predicted']))[0]

  if filename == '':
    print(f'\n==================     {msg}    ================\n')
    print(f'Score as calculated for the leader board (っಠ‿ಠ)っ {100*score:.2f}')
    print(f"Accuracy Score : {100 * accuracy_score(confusion_df['y_Actual'], confusion_df['y_Predicted']):2.f}")
    print(classification_report(confusion_df['y_Actual'], confusion_df['y_Predicted'], target_names=classes))
  else:
    print(f'\n==================     {msg}    ================\n', file = open(filename, 'w'))
    print(f'Score as calculated for the leader board (っಠ‿ಠ)っ {100*score:.2f}', file = open(filename, 'a'))
    print(f"Accuracy Score : {100 * accuracy_score(confusion_df['y_Actual'], confusion_df['y_Predicted']):2.f}", file = open(filename, 'a'))
    print(classification_report(confusion_df['y_Actual'], confusion_df['y_Predicted'], target_names=classes), file = open(filename, 'a'))    

# Heat map : show the confusion matrix
def heat_map(y_pred, y_test, classes, title='', filename=''): 
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_counts = ["{:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) \
                         for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2 in \
              zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(len(classes),len(classes))
    
    cf_matrix = pd.DataFrame(cf_matrix, columns=classes, index=classes)
    
    plt.figure(figsize = (len(classes),len(classes)))
    plt.title(title)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    
    if filename == '':
      plt.show()
    else:
      if os.path.exists(filename):
          os.remove(filename)
      plt.savefig(filename)

# predict models
def predict_models(X_test_orig, y_test, classification_models, classification_types, dst_path, save_report, save_plot, verbose):
   
  if save_plot:
    if not os.path.exists(os.path.join(dst_path, 'plots')):
      os.mkdir(os.path.join(dst_path, 'plots'))
  
  if save_report:
    if not os.path.exists(os.path.join(dst_path, 'reports')):
      os.mkdir(os.path.join(dst_path, 'reports'))

  # Load models
  for class_type in classification_types:
    # Set models dictionary
    for md_label in classification_models:
      model, scaler, le, features, _ = load_model_func(md_label, class_type, dst_path, verbose)

      # Label encoding
      _, y_test = label_encoding(y_test, le)

      # Data normalization
      _, X_test = data_normalization(X_test_orig[features], scaler)

      # predict model on the data test
      y_pred = model.predict(X_test[features])

      # Compute Accuracy Classification report
      accuracy_classification_report(y_test, y_pred, le.classes_, msg = 'Accuracy classification report for model {md_label} / class type {class_type}', 
                                      filename=os.path.join(dst_path, 'reports', f'ML_report_{md_label.upper()}_{class_type.upper()}') if save_report else '')

      # Compute heat map for the ML classification
      heat_map(y_pred, y_test, le.classes_, title='Heat map for model {md_label} / class type {class_type}', 
                                      filename=os.path.join(dst_path, 'plots', f'ML_heatmap_{md_label.upper()}_{class_type.upper()}') if save_plot else '')

# MAIN
if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument("-f", "--filename", required=True, type=str, 
                        help='Basename of the file input data without extension')
    parser.add_argument("-dst", "--process-output", required=False, type=str, default='data/process/ml_classification',
                        help='Path to save or to get the preprocessed data, plots and reports. default: data/process/ml_classification')
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
                        action='store_true', default=False, help='Save model')
    parser.add_argument("-dms", "--models_saved", required=False, type=str, default='data/models_saved',
                        help='Path to save models. default: data/models_saved')
    parser.add_argument("-sp", "--save-plots", required=False,
                        action='store_true', default=False, help='Save heatmap plots')
    parser.add_argument("-ss", "--save-reports", required=False,
                        action='store_true', default=False, help='Save report')
    parser.add_argument("-v", "--verbose", required=False,
                        action='store_true', default=False, help='Verbose')

    args = parser.parse_args()
    print(args)

    src_directory         = os.path.dirname(args.filename)
    file_basename         = os.path.splitext(os.path.basename(args.filename))[0]
    process_output        = args.process_output
    models_saved          = args.models_saved
    classification_step   = args.classification_step
    classification_types  = ['PLANTS', 'HEALTHY', 'PLANTS_DESEASES'] if args.classification_type.upper() == 'ALL' else args.classification_type.split(',')
    classification_models = ['XGB', 'ETC', 'RFC'] if args.classification_models.upper() == 'ALL' else args.classification_models.split(',')
    normalize_type        = args.normalize_type
    save_data             = args.save_data
    save_model            = args.save_model
    save_plots            = args.save_plots
    save_reports          = args.save_reports
    verbose               = args.verbose

    random.seed(42)

    if not os.path.exists(process_output):
      os.makedirs(process_output)
      local_print(f'\nInfo : folder {process_output} was created successfully !!\n', verbose)

    if classification_step in ['ALL', 'LOAD_DATA'] :
      if not os.path.exists(os.path.join(src_directory, f'{file_basename}.h5')):
        print(f'Error : filename ({file_basename}.h5) not a h5 file or doesn\'t exist')
        exit(1)
      else:
        df_features, options_dataset = load_data_from_h5(src_directory, f'{file_basename}.h5', verbose)
        
        if save_data:
          if not os.path.exists(process_output):
            print(f"Error : the path to save data '{process_output}' should exist")
            exit(1)
          if not os.path.exists(os.path.join(process_output, 'data')):
            os.mkdir(os.path.join(process_output, 'data'))

          df_features.to_pickle(os.path.join(process_output, 'data', file_basename+".pkl"))
          joblib.dump(options_dataset, os.path.join(process_output, file_basename+"_options_dataset.joblib"))
        
        if classification_step == 'LOAD_DATA':
          local_print('Info : end of the load data job')
          exit(0)

    # load data if not yet loaded
    if classification_step in ['FIT_MODEL', 'PREDICT_MODEL', 'FIT_PREDICT_MODEL'] :
        if not os.path.exists(os.path.join(process_output, file_basename+".pkl")):
            print(f'Error : the file {file_basename}.pkl does not exist')
            exit(1)
        df_features = pd.read_pickle(os.path.join(process_output, file_basename+".pkl"))
        options_dataset =  joblib.load(os.path.join(process_output, file_basename+"_options_dataset.joblib"))

    # Split data
    X_train, y_train, X_test, y_test = split_data(df_features)

    if classification_step != 'PREDICT_MODEL':
      # fit models
      fit_models(X_train, y_train, classification_models, classification_types, save_model, options_dataset, verbose)
      if classification_step == 'FIT_MODEL':
          local_print('Info : end of the models fit', verbose)
          exit(0)

    # Check if the process output exists
    if save_plots or save_reports:
      if not os.path.exists(process_output):
        print(f"Error : the path to save data '{process_output}' should exist")
        exit(1)

    # Predict model
    predict_models(X_test, y_test, classification_models, classification_types, process_output, save_reports, save_plots, verbose)
    exit(0)
