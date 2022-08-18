'''
  CLI used to manage ML classification.
'''
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

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
                'NORM_MINMAX'        : MinMaxScaler(), 
                'NORM_STANDARSCALER' : StandardScaler()
                }

def set_models_dict(classType, classModels, verbose):
  xgc = xgb.XGBClassifier(use_label_encoder=False, objective='binary:logistic' if classType == 'HEALTHY' else 'multi:softmax', eval_metric=f1_score, n_estimators=500, n_jobs=-1, verbosity=1*verbose)
  rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, verbose=verbose)
  etc = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, verbose=verbose)
  _dict = dict([('xgc', xgc), ('rfc', rfc), ('etc', etc)])
  return { md: _dict[md.lower()] for md in classModels}

def split_data(df):
    features = [f for f in df.columns if f not in ['classes','split']]
    X_train = df.loc[df.split=='train'][features]
    y_train = df.loc[df.split=='train'][['classes']]
    X_test  = df.loc[df.split=='test'][features]
    y_test  = df.loc[df.split=='test'][['classes']]
    return X_train, y_train, X_test, y_test

# Datat normalization
def data_normalization(df, scaler, save=False, fit=False):
  # fit data normalization
  if fit:
    scaler.fit(df)

  # transform data normalization
  features = df.columns
  df[features] = scaler.transform(df).astype(np.float32)

  return scaler, df

# Label encoding
def label_encoding(df, le, class_type, fit=False):
  _df = df.copy()
  # Class agregation
  if class_type == 'HEALTHY':
    _df['classes'] = _df.classes.apply(lambda c: 'healthy' if 'healthy' in c.lower() else 'not_healthy')
  if class_type == 'PLANTS':
    _df['classes'] = _df.classes.apply(lambda c: c.split('_')[0])

  # fit label encoding  
  if fit:
    le.fit(_df['classes'])

  # transform label encoding
  _df['classes'] = le.transform(_df['classes'])

  return le, _df

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
    local_print(f'\033[93mInfo : model {md_label} has been saved !!!§\033[0m', verbose)

# Load model
def load_model_func(md_label, class_type, dst_path, verbose):
    path = os.path.join(dst_path, f'ML_{md_label.upper()}_{class_type}.pkl.z')
    if not os.path.exists(path):
        print(f"\033[91mError : the ML_{md_label.upper()}_{class_type}.pkl.z does not exist !!!\033[0m")
        exit()

    md_dict = joblib.load(path)
    model = md_dict['ml_model']
    scaler = md_dict['ml_scaler']
    le = md_dict['ml_label_encoder']
    features = md_dict['ml_features']
    options_dataset = md_dict['options_dataset']
    local_print(f'\033[93mInfo : model {md_label} has been loaded !!!§\033[0m', verbose)

    return model, scaler, le, features, options_dataset

# fit models
def fit_models(X_train, y_orig, classification_models, classification_types, options_dataset, save ,md_dst, verbose):
  # Data normalization
  scaler, X_train = data_normalization(X_train, scaler_dict[normalize_type], save=save, fit=True)

  # Processing
  models_dict = {}
  for class_type in classification_types:
    # Label encoding
    le, yl_train = label_encoding(y_orig, LabelEncoder(), class_type, fit=True)

    # Set models dictionary
    models_dict[class_type] = set_models_dict(class_type, classification_models, verbose)

    # Fit models
    for md_label in models_dict[class_type].keys():
        local_print(f'\n\033[92m+++++++++++      Fitting for model {md_label} and class type {class_type} started     ++++++++++\033[0m\n', verbose)
        models_dict[class_type][md_label].fit(X_train, yl_train.classes)
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
    print(f'\033[96mSpearmnr score (っಠ‿ಠ)っ\t{100*score:.2f}')
    print(f"Accuracy Score         :\t{100 * accuracy_score(confusion_df['y_Actual'], confusion_df['y_Predicted']):.2f}")
    print(f"{classification_report(confusion_df['y_Actual'], confusion_df['y_Predicted'], target_names=classes)}\033[0m")
  else:
    print(f'\n==================     {msg}    ================\n', file = open(filename, 'w'))
    print(f'Spearmnr score (っಠ‿ಠ)っ\t{100*score:.2f}', file = open(filename, 'a'))
    print(f"Accuracy Score         :\t{100 * accuracy_score(confusion_df['y_Actual'], confusion_df['y_Predicted']):.2f}", file = open(filename, 'a'))
    print(classification_report(confusion_df['y_Actual'], confusion_df['y_Predicted'], target_names=classes, zero_division=0), file = open(filename, 'a'))    

# Heat map : show the confusion matrix
def heat_map(y_pred, y_test, classes, title='', filename=''): 
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_counts = ["{:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) \
                         for value in (cf_matrix.T/cf_matrix.T.sum(axis=0)).T.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in \
              zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(len(classes),len(classes))
    
    cf_matrix = pd.DataFrame(cf_matrix, columns=classes, index=classes)
    
    fig_size = 6 if len(classes) < 4 else 9 if len(classes) < 16 else 18
    fnt_size = 9 if len(classes) < 4 else 8
    y_pos    = 1.2 if len(classes) < 4 else 1.06 
    
    fig = plt.figure("HeatMap Plot", figsize = (fig_size, fig_size*9//16))
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', annot_kws={"size": fnt_size-3})
    fig.suptitle(title, fontsize=fnt_size*1.5, color='#C86400', y=y_pos)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, fontsize=fnt_size, color='grey', ha = 'left')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fnt_size, color='grey')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    if filename == '':
      plt.show()
    else:
      if os.path.exists(filename):
          os.remove(filename)
      plt.savefig(filename, bbox_inches='tight', orientation='landscape')
      plt.close()

# predict models
def predict_models(X_test_orig, y_test, classification_models, classification_types, md_dst, dst_path, save_report, save_plot, verbose):
   
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
      local_print(f'\n\033[92m+++++++++++      Prediction for model {md_label} and class type {class_type} started     ++++++++++\033[0m\n', verbose)
      model, scaler, le, features, _ = load_model_func(md_label, class_type, md_dst, verbose)

      # Label encoding
      _, yl_test = label_encoding(y_test, le, class_type)

      # Data normalization
      _, X_test = data_normalization(X_test_orig[features], scaler)

      # predict model on the data test
      y_pred = model.predict(X_test[features])

      # Compute Accuracy Classification report
      accuracy_classification_report(yl_test.classes, y_pred, le.classes_, msg = f'Accuracy classification report for model {md_label} and class type {class_type}', 
                                      filename=os.path.join(dst_path, 'reports', f'ML_report_{md_label.upper()}_{class_type.upper()}.txt') if save_report else '')

      # Compute heat map for the ML classification
      heat_map(y_pred, yl_test.classes, le.classes_, title=f'Heatmap for model {md_label} and class type {class_type}', 
                                      filename=os.path.join(dst_path, 'plots', f'ML_heatmap_{md_label.upper()}_{class_type.upper()}') if save_plot else '')

# MAIN
if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument("-cs", "--classification-step", required=False, type=str, default="ALL",
                        help='Classification step: LOAD_DATA, FIT_MODEL, PREDICT_MODEL, FIT_PREDICT_MODEL, ALL (default)')
    parser.add_argument("-f", "--filename", required=True, type=str, 
                        help='Basename of the file input data without extension')
    parser.add_argument("-dst", "--process-output", required=False, type=str, default='data/process/ml_classification',
                        help='Path to save or to get the preprocessed data, plots and reports. default: data/process/ml_classification')
    parser.add_argument("-sd", "--save-data", required=False,
                        action='store_true', default=True, help='Save options_datasets json file and converted data from h5 format to DataFrame one with flag train/test flag')
    parser.add_argument("-nortype", "--normalize-type", required=False, type=str, default='NORM_MINMAX',
                        help='Normalize data (NORM_STANDARSCALER or NORM_MINMAX normalization) (Default: NORM_MINMAX)')
    parser.add_argument("-cm", "--classification-models", required=False, type=str, default="ALL",
                        help='Classification models: XGC, ETC, RFC, ALL (default). Example -cm=RFC,ETC')
    parser.add_argument("-ct", "--classification-types", required=False, type=str, default="ALL",
                        help='Classification type: PLANTS, HEALTHY, PLANTS_DESEASES classes, ALL (default)')
    parser.add_argument("-sm", "--save-model", required=False,
                        action='store_true', default=True, help='Save model')
    parser.add_argument("-dms", "--models_saved", required=False, type=str, default='data/models_saved',
                        help='Path to save models. default: data/models_saved')
    parser.add_argument("-sp", "--save-plots", required=False,
                        action='store_true', default=False, help='Save heatmap plots')
    parser.add_argument("-sr", "--save-reports", required=False,
                        action='store_true', default=False, help='Save report')
    parser.add_argument("-v", "--verbose", required=False,
                        action='store_true', default=False, help='Verbose')

    args = parser.parse_args()

    # Function to check if arguments are into the list
    def list_check(arg, ref):
      arg = [a.upper() for a in arg.split(',')]
      if 'ALL' in arg:
        return ref
      elif len([a for a in arg if a in ref]) > 0:
        return [a for a in arg if a in ref]
      else:
        return ref[-1:]

    classification_step   = args.classification_step.upper()
    src_directory         = os.path.dirname(args.filename)
    file_basename         = os.path.splitext(os.path.basename(args.filename))[0]
    process_output        = args.process_output
    models_saved          = args.models_saved
    classification_types  = list_check(args.classification_types, ['HEALTHY', 'PLANTS', 'PLANTS_DESEASES'])
    classification_models = list_check(args.classification_models, ['XGC', 'RFC', 'ETC'])
    normalize_type        = args.normalize_type
    save_data             = args.save_data
    save_model            = args.save_model
    save_plots            = args.save_plots
    save_reports          = args.save_reports
    verbose               = args.verbose
    
    random.seed(42)

    # Display the arguments set for the programm
    os.system('clear')
    print("\n===========    Script arguemnts    ===========\n")
    for k in ['src_directory','file_basename']+[k for k in vars(args).keys() if k !='filename']:
      print(f'[+] {k.ljust(22)}: {vars()[k]}')

    # Check if the classification step is correct
    if classification_step not in ['LOAD_DATA', 'FIT_MODEL', 'PREDICT_MODEL', 'FIT_PREDICT_MODEL', 'ALL']:
      print(f"\n\033[91mError : the classification step ({classification_step}) is not correct; please check your arguemnts !!!\033[0m")
      exit(1)

    if not os.path.exists(process_output):
      os.makedirs(process_output)
      local_print(f'\n\033[93mInfo : folder {process_output} was created successfully !!\033[0m\n', verbose)

    if classification_step in ['ALL', 'LOAD_DATA'] :
      if not os.path.exists(os.path.join(src_directory, f'{file_basename}.h5')):
        print(f'\033[91mError : filename ({file_basename}.h5) not a h5 file or doesn\'t exist\033[0m')
        exit(1)
      else:
        df_features, options_dataset = load_data_from_h5(src_directory, f'{file_basename}.h5', verbose)

        # Fill NaN values
        df_features.fillna(0, inplace=True)
        
        if save_data:
          if not os.path.exists(process_output):
            print(f"\033[91mError : the path to save data '{process_output}' should exist\033[0m")
            exit(1)
          if not os.path.exists(os.path.join(process_output, 'data')):
            os.mkdir(os.path.join(process_output, 'data'))

          df_features.to_pickle(os.path.join(process_output, 'data', file_basename+".pkl"))
          joblib.dump(options_dataset, os.path.join(process_output, 'data', file_basename+"_options_dataset.joblib"))
        
        if classification_step == 'LOAD_DATA':
          local_print('\033[93mInfo : Load data job ended\033[0m', verbose)
          exit(0)

    # load data if not yet loaded
    if classification_step in ['FIT_MODEL', 'PREDICT_MODEL', 'FIT_PREDICT_MODEL'] :
        if not os.path.exists(os.path.join(process_output, 'data', file_basename+".pkl")):
            print(f'\033[91mError : the file {file_basename}.pkl does not exist\033[0m')
            exit(1)
        df_features = pd.read_pickle(os.path.join(process_output, 'data', file_basename+".pkl"))
        options_dataset =  joblib.load(os.path.join(process_output, 'data', file_basename+"_options_dataset.joblib"))

    # Split data
    X_train, y_train, X_test, y_test = split_data(df_features)

    if classification_step != 'PREDICT_MODEL':
      # fit models
      fit_models(X_train, y_train, classification_models, classification_types, options_dataset, save_model, models_saved, verbose)
      if classification_step == 'FIT_MODEL':
          local_print('\033[93mInfo : Models fit step ended\033[0m', verbose)
          exit(0)

    # Check if the process output exists
    if save_plots or save_reports:
      if not os.path.exists(process_output):
        print(f"\033[91mError : the path to save data '{process_output}' should exist\033[0m")
        exit(1)

    # Predict model
    predict_models(X_test, y_test, classification_models, classification_types, models_saved, process_output, save_reports, save_plots, verbose)
    local_print(f"\033[93mInfo : {'Prediction step' if classification_step == 'PREDICT_MODEL' else 'Classification job'} ended\033[0m", verbose)
    exit(0)
