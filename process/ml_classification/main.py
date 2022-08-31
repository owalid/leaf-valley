'''
  CLI used to manage ML classification.
'''
from datetime import datetime
from fileinput import filename
import os
import sys
import random
import joblib
from datetime import datetime as dt

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
import argparse as ap
from inspect import getsourcefile

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, label_binarize

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from scipy import stats
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score, \
                            precision_score, recall_score, log_loss, average_precision_score, auc, roc_curve

from sklearn_evaluation.plot.roc import roc
from sklearn_evaluation.plot.precision_recall import precision_recall
from sklearn_evaluation.plot.classification import feature_importances


current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, os.path.sep.join(current_dir.split(os.path.sep)[:-2]))

from load_data_from_h5 import load_data_from_h5

from pathlib import Path

VERBOSE = False
scaler_dict = {
                'NORM_MINMAX'        : MinMaxScaler(), 
                'NORM_STANDARSCALER' : StandardScaler()
                }

def local_print(msg):
    if VERBOSE:
        print(msg)

def set_models_dict(classType, classModels):
  xgc = xgb.XGBClassifier(use_label_encoder=False, eval_metric=['error','logloss','auc'], objective='binary:logistic' if classType == 'HEALTHY' else 'multi:softproba', 
                          n_estimators=500, n_jobs=-1)
  rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, verbose=VERBOSE)
  etc = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, verbose=VERBOSE)
  _dict = dict([('xgc', xgc), ('rfc', rfc), ('etc', etc)])
  return { md: _dict[md.lower()] for md in classModels}

def create_mlruns_folder(dir, run_name=""):
    mlflow.set_tracking_uri(os.path.join(os.getcwd(), dir, "mlruns"))
    mlflow.set_experiment(run_name)
    try:
        # creating a new experiment
        exp_id = mlflow.create_experiment(name=run_name, artifact_location=Path.cwd().joinpath(dir, "mlruns").as_uri())
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(run_name).experiment_id

    return exp_id

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
def save_model_func(md_label, model, scaler, le, features, class_type, options_dataset, dst_path):
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
    local_print(f'\033[93mInfo : model {md_label} has been saved !!!§\033[0m')

# Load model
def load_model_func(md_label, class_type, dst_path):
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
    local_print(f'\033[93mInfo : model {md_label} has been loaded !!!§\033[0m')

    return model, scaler, le, features, options_dataset

# XGC Learning Curve function
def xgc_learning_curve(model, filename):
    results = model.evals_result()
    _, ax = plt.subplots(1,3,figsize=(12,3))
    i = 0
    for eval_metric in model.get_params()['eval_metric']:
      ax[i].plot(results['validation_0'][eval_metric], label='train')
      ax[i].plot(results['validation_1'][eval_metric], label='test')
      ax[i].title.set_text(eval_metric)
      ax[i].legend()
      i+=1
    plt.savefig(filename, bbox_inches='tight', orientation='landscape')
    plt.close()     

# Cross validation learning curve
def plot_learning_curve(estimator, X, y, title, filename, ylim=(0.7, 1.01), cv=ShuffleSplit(n_splits=3, test_size=0.3, random_state=0), 
                        n_jobs=-1, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, 
                      the training samples vs fit times curve, 
                      the fit times vs score curve.
    """
    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training obs")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    fit_times_mean    = np.mean(fit_times, axis=1)
    fit_times_std     = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].set_title("Scalability of the model")
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].set_title("Performance of the model")
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(fit_time_sorted, test_scores_mean_sorted - test_scores_std_sorted, test_scores_mean_sorted + test_scores_std_sorted, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")

    plt.savefig(filename, bbox_inches='tight', orientation='landscape')
    plt.close()

# fit models
def fit_models(X_train, y_orig, classification_models, classification_types, options_dataset, save ,md_dst, dst_path, dtest, expid):
  # Data normalization
  scaler, X_train = data_normalization(X_train, scaler_dict[normalize_type], save=save, fit=True)

  # Processing
  models_dict = {}
  for class_type in classification_types:
    # Label encoding
    le, yl_train = label_encoding(y_orig, LabelEncoder(), class_type, fit=True)
    _, yl_test = label_encoding(dtest[1], le, class_type)

    # Set models dictionary
    models_dict[class_type] = set_models_dict(class_type, classification_models)

    # Fit models
    for md_label in models_dict[class_type].keys():
        # create a training output file if not exist
        if not os.path.exists(os.path.join(dst_path, 'training', f'{md_label.upper()}_{class_type.upper()}')):
          os.makedirs(os.path.join(dst_path, 'training', f'{md_label.upper()}_{class_type.upper()}'))

        # create a subfolder
        if not os.path.exists(os.path.join(dst_path, 'training', f'{md_label.upper()}_{class_type.upper()}')):
          os.mkdir(os.path.join(dst_path, 'training', f'{md_label.upper()}_{class_type.upper()}'))

        filename = os.path.join(dst_path, 'training', f'{md_label.upper()}_{class_type.upper()}', f'Training_Plot_EvalPerfs_{md_label.upper()}_{class_type.upper()}.jpeg')
        if os.path.exists(filename):
            os.remove(filename)

        with mlflow.start_run(experiment_id=expid) as r:
            mlflow.set_tags({"mlflow.runName": f'MLC Training {md_label.upper()} {class_type.upper()}',
                            'mlflow.note.content': f'This is the output of the ML CLassification model training : model ({md_label}) and class type ({class_type})'})
            
            start = dt.now()
            local_print(f'\033[92m+++++++++++      Fitting for model {md_label} and class type {class_type} started     ++++++++++\033[0m\n')
            if md_label == 'xgc':
              models_dict[class_type][md_label].fit(X_train, yl_train.classes, eval_set=[(X_train,yl_train.classes),(X_test,yl_test.classes)], 
                                                    verbose=max(1,models_dict[class_type][md_label].get_params()['n_estimators']//10 + 1)*10)
              # plot learning curves
              xgc_learning_curve(models_dict[class_type][md_label], filename)

            else:
              models_dict[class_type][md_label].fit(X_train, yl_train.classes)
              
            # plot learning curve
            plot_learning_curve(models_dict[class_type][md_label], X_train, yl_train.classes, f"Learning curve for the {md_label} model and class type {class_type}", 
                                filename, ylim=(0.7, 1.01), cv=ShuffleSplit(n_splits=3, test_size=0.2, random_state=42), 
                                n_jobs=-1, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 5))

            if save:
              if not os.path.exists(md_dst):
                os.makedirs(md_dst)              
              save_model_func( md_label, models_dict[class_type][md_label], scaler, le, X_train.columns, class_type,options_dataset, md_dst)
            local_print(f'\033[93mInfo : The training of the models finished, it took {dt.now() - start}\033[0m\n')

            mlflow.set_tags({'ProblemType': 'ML Classification', 
                            'ModelType': md_label, 
                            'ModelLibrary': 'XGBoost' if md_label=='xgc' else 'Scikit-Learn'})

            mlflow.log_params(models_dict[class_type][md_label].get_params(deep=False))

            mlflow.log_artifacts(os.path.join(dst_path, 'training', f'{md_label.upper()}_{class_type.upper()}'))

            mlflow.end_run()

# Accuracy Classification Report
def accuracy_classification_report(y_test, y_pred, y_score, classes, msg = '', filename=''):
  confusion_mtx = {
                    'y_Actual': np.array(y_test),
                    'y_Predicted': y_pred
                }

  confusion_df = pd.DataFrame(confusion_mtx, columns=['y_Actual','y_Predicted'])    

  metrics = {}

  metrics['spearmanr']       = 100 * (stats.spearmanr(confusion_df['y_Actual'], confusion_df['y_Predicted']))[0]
  metrics['accuracy']        = 100 * accuracy_score(confusion_df['y_Actual'], confusion_df['y_Predicted'])
  if len(np.unique(confusion_df['y_Actual']))== 2:
    metrics['auc score']       = 100 * roc_auc_score(confusion_df['y_Actual'], y_pred, average='macro')
  else:
    metrics['auc score']       = 100 * roc_auc_score(confusion_df['y_Actual'], y_score, multi_class='ovr', average='macro')
  metrics['log loss']        = 100 * log_loss(confusion_df['y_Actual'], y_score)
  metrics['precision score'] = 100 * precision_score(confusion_df['y_Actual'], confusion_df['y_Predicted'], average='macro')
  metrics['recall score']    = 100 * recall_score(confusion_df['y_Actual'], confusion_df['y_Predicted'], average='macro')
  metrics['f1 score']        = 100 * f1_score(confusion_df['y_Actual'], confusion_df['y_Predicted'], average='macro')

  output_metrics  = f"Spearmnr score (っಠ‿ಠ)っ\t{metrics['spearmanr']:.2f}\n"
  output_metrics += f"Accuracy Score         :\t{metrics['accuracy']:.2f}\n"
  output_metrics += f"auc score              :\t{metrics['auc score']:.2f}\n"
  output_metrics += f"log loss               :\t{metrics['log loss']:.2f}\n"
  output_metrics += f"precision score        :\t{metrics['precision score']:.2f}\n"
  output_metrics += f"recall score           :\t{metrics['recall score']:.2f}\n"
  output_metrics += f"f1 score               :\t{metrics['f1 score']:.2f}\n\n"
  output_metrics += f"{classification_report(confusion_df['y_Actual'], confusion_df['y_Predicted'], target_names=classes)}"

  if filename == '':
    print(f'\n==================     {msg}    ================\n\n\033[96m{output_metrics}\033[0m')
  else:
    print(f'\n==================     {msg}    ================\n\n{output_metrics}', file = open(filename, 'w'))

  return metrics

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
      plt.savefig(filename, bbox_inches='tight', orientation='landscape')
      plt.close()

# Precision Recal and Feature Importances curves
def prec_recal_roc_curves(est, y_test, y_score, features_top_n, filename=''):
    _, ax = plt.subplots(1,3, figsize=(16,3))
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6

    ax[0] = precision_recall(y_test, y_score, ax=ax[0])
    ax[1] = roc(y_test, y_score, ax=ax[1])
    ax[2] = feature_importances(est, top_n=features_top_n, feature_names=est.feature_names_in_.tolist(), ax=ax[2])

    # Compute the area under curve
    n_classes = len(np.unique(y_test))

    if n_classes > 2:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        avg_prec = average_precision_score(y_test_bin, y_score, average="micro")
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    else:
        avg_prec = average_precision_score(y_test, y_score[:,1], average="micro")
        fpr, tpr, _ = roc_curve(y_test, y_score[:,1])

    roc_auc = auc(fpr, tpr)

    ax[0].set_xlim([-0.05, 1.05]), ax[0].set_ylim([-0.05, 1.05]), ax[0].set_title(f'Precision-Recall\n (area = {avg_prec:.2f})')
    ax[1].set_xlim([-0.05, 1.05]), ax[1].set_ylim([-0.05, 1.05]), ax[1].set_title(f'ROC\n (area = {roc_auc:.2f})')
    ax[2].set_title(f'The {features_top_n} importance\nfeatures')
    for i in range(2):
        ax[i].legend().remove()


    plt.subplots_adjust(wspace=.8)

    if filename == '':
      plt.show()
    else:
      plt.savefig(filename, bbox_inches='tight', orientation='landscape')
      plt.close()

# predict models
def predict_models(X_test_orig, y_test, classification_models, classification_types, md_dst, dst_path, expid):
  
  # Load models
  for class_type in classification_types:
    # Set models dictionary
    for md_label in classification_models:
      local_print(f'\033[92m+++++++++++      Prediction for model {md_label} and class type {class_type} started     ++++++++++\033[0m\n')
      model, scaler, le, features, _ = load_model_func(md_label, class_type, md_dst)

      # Label encoding
      _, yl_test = label_encoding(y_test, le, class_type)

      # Data normalization
      _, X_test = data_normalization(X_test_orig[features], scaler)

      # predict model on the data test
      y_pred  = model.predict(X_test[features])
      y_score = model.predict_proba(X_test[features])

      # create a subfolder
      if not os.path.exists(os.path.join(dst_path, 'plots_reports', f'{md_label.upper()}_{class_type.upper()}')):
        os.makedirs(os.path.join(dst_path, 'plots_reports', f'{md_label.upper()}_{class_type.upper()}'))

      with mlflow.start_run(experiment_id=expid) as r:
          mlflow.set_tags({"mlflow.runName": f'MLC Testing {md_label.upper()} {class_type.upper()}',
                           'mlflow.note.content': f'This is the output of the test for the ML CLassification model : model ({md_label}) and class type ({class_type})'})

          # Compute Accuracy Classification report
          metrics = accuracy_classification_report(yl_test.classes, y_pred, y_score, le.classes_, msg = f'Accuracy classification report for model {md_label} and class type {class_type}', 
                                          filename=os.path.join(dst_path, 'plots_reports', f'{md_label.upper()}_{class_type.upper()}', f'ML_report_{md_label.upper()}_{class_type.upper()}.txt'))

          # Compute heat map for the ML classification
          heat_map(y_pred, yl_test.classes, le.classes_, title=f'Heatmap for model {md_label} and class type {class_type}', 
                                          filename=os.path.join(dst_path, 'plots_reports', f'{md_label.upper()}_{class_type.upper()}', f'ML_heatmap_{md_label.upper()}_{class_type.upper()}.png'))


          # Model performances
          prec_recal_roc_curves(model, yl_test.classes, y_score, 20,
                                filename=os.path.join(dst_path, 'plots_reports', f'{md_label.upper()}_{class_type.upper()}', f'ML_model_perfs_{md_label.upper()}_{class_type.upper()}.png'))


          mlflow.set_tags({'ProblemType': 'ML Classification', 
                           'ModelType': md_label, 
                           'ModelLibrary': 'XGBoost' if md_label=='xgc' else 'Scikit-Learn'})

          mlflow.log_params(model.get_params(deep=False))
          mlflow.log_metrics(metrics)

          mlflow.log_artifacts(os.path.join(dst_path, 'plots_reports', f'{md_label.upper()}_{class_type.upper()}'))

          mlflow.end_run()

# MAIN
if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument("-cs", "--classification-step", required=False, type=str, default="ALL",
                        help='Classification step: LOAD_DATA, FIT_MODEL, PREDICT_MODEL, FIT_PREDICT_MODEL, ALL (default)')
    parser.add_argument("-f", "--filename", required=True, type=str, 
                        help='path and file name of the input data')
    parser.add_argument("-dst", "--process-output", required=False, type=str, default='data/process/ml_classification',
                        help='Path to save or to get the preprocessed data, plots and reports. default: data/process/ml_classification')
    parser.add_argument("-sd", "--save-data", required=False, action='store_false', default=True, 
                        help='Save options_datasets json file and converted data from h5 format to DataFrame one with flag train/test flag, default True')
    parser.add_argument("-th", "--threshold", required=False, default=0.1, 
                        help='Threshold used for the filter method to select features')
    parser.add_argument("-nortype", "--normalize-type", required=False, type=str, default='NORM_MINMAX',
                        help='Normalize data (NORM_STANDARSCALER or NORM_MINMAX normalization) (Default: NORM_MINMAX)')
    parser.add_argument("-cm", "--classification-models", required=False, type=str, default="ALL",
                        help='Classification models: XGC, ETC, RFC, ALL (default). Example -cm=RFC,ETC')
    parser.add_argument("-ct", "--classification-types", required=False, type=str, default="ALL",
                        help='Classification type: PLANTS, HEALTHY, PLANTS_DESEASES classes, ALL (default)')
    parser.add_argument("-sm", "--save-model", required=False,
                        action='store_false', default=True, help='Save model, default True')
    parser.add_argument("-dms", "--dest-models-saved", required=False, type=str, default='data/models_saved',
                        help='Path to save models. default: data/models_saved')
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
    dest_models_saved     = args.dest_models_saved
    classification_types  = list_check(args.classification_types, ['HEALTHY', 'PLANTS', 'PLANTS_DESEASES'])
    classification_models = list_check(args.classification_models, ['XGC', 'RFC', 'ETC'])
    normalize_type        = args.normalize_type
    save_data             = args.save_data
    save_model            = args.save_model
    threshold             = float(args.threshold)
    VERBOSE               = args.verbose
    
    random.seed(42)

    # Display the arguments set for the programm
    print("\n===========    Script arguements    ===========\n")
    for k in ['src_directory','file_basename']+[k for k in vars(args).keys() if k not in ['filename', 'verbose']]:
      print(f'[+] {k.ljust(max([len(a) for a in vars(args).keys()])+2)}: {vars()[k]}')

    # Check if the classification step is correct
    if classification_step not in ['LOAD_DATA', 'FIT_MODEL', 'PREDICT_MODEL', 'FIT_PREDICT_MODEL', 'ALL']:
      print(f"\n\033[91mError : the classification step ({classification_step}) is not correct; please check your arguemnts !!!\033[0m")
      exit(1)

    if not os.path.exists(process_output):
      os.makedirs(process_output)
      local_print(f'\n\033[93mInfo : folder {process_output} was created successfully !!\033[0m\n')

    if classification_step in ['ALL', 'LOAD_DATA'] :
      if not os.path.exists(os.path.join(src_directory, f'{file_basename}.h5')):
        print(f'\033[91mError : filename ({file_basename}.h5) not a h5 file or doesn\'t exist\033[0m')
        exit(1)
      else:
        df_features, options_dataset = load_data_from_h5(src_directory, f'{file_basename}.h5', threshold, VERBOSE)

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
          local_print('\033[93mInfo : Load data job ended\033[0m')
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

    # Create a mlruns folder for the MLFLOW 
    exp_id = create_mlruns_folder(process_output, run_name="ML Classification")

    if classification_step != 'PREDICT_MODEL':
      # fit models
      fit_models(X_train, y_train, classification_models, classification_types, options_dataset, save_model, dest_models_saved, process_output, (X_test, y_test), exp_id)
      if classification_step == 'FIT_MODEL':
          local_print('\033[93mInfo : Models fit step ended\033[0m')
          exit(0)

    # Check if the process output exists
    if not os.path.exists(process_output):
      print(f"\033[91mError : the path to save data '{process_output}' should exist\033[0m")
      exit(1)

    # Predict model
    predict_models(X_test, y_test, classification_models, classification_types, dest_models_saved, process_output, exp_id)
    local_print(f"\033[93mInfo : {'Prediction step' if classification_step == 'PREDICT_MODEL' else 'Classification job'} ended\033[0m")
    exit(0)
