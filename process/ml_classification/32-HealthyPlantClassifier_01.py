# # `Healthy Plantes Classifier`

# Lib & Dependencies
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm

from itertools import repeat
import concurrent.futures

# import sweetviz as sv
from scipy import stats
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# #### 1. Load and process data
df_features = pd.read_pickle('../data/preprocess/ALL/augmentation/export/data_all_all_graycoprops_lpb_histogram_hue_moment_haralick_histogram_hsv_histogram_lab_pyfeats.pkl')


# Set label = (healthy or not)
df_features['label'] = df_features.classes.apply(lambda l: 1*('healthy' in l))


X_train, X_test, y_train, y_test = train_test_split(df_features.drop(columns=['label','classes']), df_features['label'], test_size=0.3, random_state=42)
print(df_features.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def accuracy_classification_report(y_test, preds, col):
  confusion_mtx = {
      'y_Actual': np.array(y_test),
      'y_Predicted': preds
  }

  confusion_df = pd.DataFrame(confusion_mtx, columns=['y_Actual','y_Predicted'])    

  score = (stats.spearmanr(confusion_df['y_Actual'], confusion_df['y_Predicted']))[0]

  print('Score as calculated for the leader board (っಠ‿ಠ)っ {}'.format(score))
  print('Accuracy Score :',accuracy_score(confusion_df['y_Actual'], confusion_df['y_Predicted']))
  print(classification_report(confusion_df['y_Actual'], confusion_df['y_Predicted'], target_names=col))

# Heat map : show the confusion matrix
def heat_map(preds, y_test, col, filename=''): 
    cf_matrix = confusion_matrix(y_test, preds)
    group_counts = ["{:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) \
                         for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2 in \
              zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(len(col),len(col))
    
    cf_matrix = pd.DataFrame(cf_matrix, columns=col, index=col)
    
    # plt.figure(figsize = (32,16))
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    if filename == '':
      plt.show()
    else:
      plt.savefig(filename)

# Fit the model model 
def fit_model(X_train, y_train, model_):
    print(f"======   The model used is : {model_}   ========\n")
    
    model = models[model_]
    model.fit(X_train, y_train)

    return model

# Fit and test the prediction model 
def prediction_hackathon( X_test, y_test, model, model_, col=['Unhealthy','Healthy']):
    print(f"======   The model used is : {model_}   ========\n")
    
    if not vrb_seuil:
      # preds_train = model.predict(X_train) 
      preds = model.predict(X_test)
    else:
      # preds_train = (model.predict_proba(X_train)[:,1]>=seuil).astype(int)
      preds = (model.predict_proba(X_test)[:,1]>=seuil).astype(int)


    accuracy_classification_report(y_test, preds, col)
    
    # # Accuracy of the Test data
    print(f"\n==> Accuracy Score for the test data : {100 * accuracy_score(y_test, preds):.2f}")
        
    # Spearman correlation for the test data
    print(f"\n==> The SPEARMAN CORRELATION of the test data is : {100 * stats.spearmanr(y_test, preds)[0]:.2f}")
    
    global df_accuracy
    df_accuracy.loc[model_, 'SPEARMAN CORRELATION'] = 100 * stats.spearmanr(y_test, preds)[0]
    df_accuracy.loc[model_, 'Accuracy Score']       = 100 * accuracy_score(y_test, preds)
    df_accuracy.loc[model_, 'F1 Score macro']       = 100 * f1_score(y_test, preds, average='macro')
    df_accuracy.loc[model_, 'F1 Score weighted']    = 100 * f1_score(y_test, preds, average='weighted')
    
    # Heat Map
    heat_map(preds, y_test, col, filename=f'plots/heatmape_healthy_{model_}.png')


# xgb.set_config(verbosity=3)
xgc = xgb.XGBClassifier(use_label_encoder=False, objective='binary:logistic', eval_metric=f1_score, n_estimators=3000, n_jobs=-1)
rfc = RandomForestClassifier(n_estimators=3000, n_jobs=-1, verbose=0)
etc = ExtraTreesClassifier(n_estimators=3000, n_jobs=-1, verbose=0)

seuil = 0.5
vrb_seuil = False
model_target = {}
df_accuracy  = pd.DataFrame()

models = dict([('rfc', rfc), ('etc', etc), ('xgc', xgc)])

for md_ in models.keys():
    model_target[md_] = fit_model(X_train, y_train, md_)
    prediction_hackathon( X_test, y_test, model_target[md_], md_)

print(df_accuracy)

for degree in range(2,13):
    svcp = SVC(kernel='poly', degree=degree, random_state=42) # Polynomial Kernel
    models = dict([(f'svcp{degree}', svcp)])
    model_target[md_] = fit_model(X_train, y_train, md_)
    prediction_hackathon( X_test, y_test, model_target[md_], md_)

print(df_accuracy)