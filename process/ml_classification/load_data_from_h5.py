''' 
    convert h5 file data and prepare it for ML classification 
    input  : file h5
    output : DataFrame
             options dataset used for the features creation
'''

import h5py
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
import concurrent.futures

import os
import sys
from inspect import getsourcefile
current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, os.path.sep.join(current_dir.split(os.path.sep)[:-2]))

from utilities.utils import local_print

def convert_to_df(lst):
    global hf
    df = pd.DataFrame()
    for f in lst:
        df[f] = list(hf[f])
    return df

def load_data_from_h5(path, file, verbose):
    start = dt.now()
    local_print(f'Job to convert h5 file to DataFrame started at : {start}', verbose)

    basename = os.path.basename(file).split('.')[0]

    hf = h5py.File(os.path.join(path, basename+".h5"), 'r')

    # Get the options dataset values and create a json file for them
    options_dataset = json.loads(list(hf.get('options_dataset'))[0].decode("utf-8"))

    features = [f for f in hf.keys() if f != 'options_dataset']
    features_lst = []
    features.sort()

    for i in range((len(features) // 25 + 1)):
        features_lst.append(features[25*i:25*(i+1)])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(convert_to_df, features_lst), total=len(features_lst)))

    hf.close()

    df_features = pd.DataFrame()
    for df in results:
        df_features = pd.concat([df_features, df], axis=1)

    df_features['classes'] = df_features.classes.apply(lambda l: l.decode("utf-8"))
    df_features = df_features.copy()

    local_print(f'Info : DataFrame shape : {df_features.shape}', verbose)
    local_print(f'Job to convert h5 file to DataFrame ended at : {start}, \tIt took : {dt.now() - start} s', verbose)

    return df_features, options_dataset