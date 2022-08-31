''' 
    convert h5 file data and prepare it for ML classification 
    input  : file h5
    output : DataFrame
             options dataset used for the features creation
'''

import os
from timeit import repeat
import h5py
import json
import random
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from itertools import repeat
from sklearn.utils import shuffle
from datetime import datetime as dt
from sklearn.feature_selection import VarianceThreshold


def local_print(msg, verbose):
    if verbose:
        print(msg)
 
def convert_to_df(lst, filename):
    with h5py.File(filename, 'r') as hf:
        try:
            df = pd.DataFrame()
            for f in lst:
                df[f] = list(hf[f])
            return df
        except KeyError as e:
            print(e)
        finally:
           hf.close()
           

def load_data_from_h5(path, file, threshold, verbose):
    try:
        start = dt.now()
        local_print(f'Job to convert h5 file to DataFrame started at : {start}', verbose)

        basename = os.path.basename(file).split('.')[0]

        hf = h5py.File(os.path.join(path, basename+".h5"), 'r')

        # Get the options dataset values and create a json file for them
        options_dataset = json.loads(list(hf.get('options_dataset'))[0].decode("utf-8"))

        features = [f for f in hf.keys() if f != 'options_dataset']
        features_lst = []
        features.sort()
        hf.close()

        step = 25
        for i in range((len(features) // step + 1)):
            features_lst.append(features[step*i:step*(i+1)])

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(convert_to_df, features_lst, repeat(os.path.join(path, basename+".h5"))), total=len(features_lst)))

        df_features = pd.DataFrame()
        for df in results:
            df_features = pd.concat([df_features, df], axis=1)

        df_features['classes'] = df_features.classes.apply(lambda l: l.decode("utf-8"))

        # Feature selection : apply a filter method 
        sel = VarianceThreshold(threshold=threshold)
        sel.fit(df_features[[f for f in df.columns if f !='classes']])
        df_features = df_features[['classes'] + sel.get_feature_names_out().tolist()].copy()

        # Drop duplicate rows
        df_features.drop_duplicates(inplace=True)

        # SPlit manually data into train/test
        df_features = shuffle(df_features)

        df_features.loc[random.sample(df_features.index.to_list(), int(.7*len(df_features))),'split'] = 'train'
        df_features['split'].fillna('test', inplace=True)

        df_features = df_features.copy()

        local_print(f'Info : DataFrame shape : {df_features.shape}', verbose)
        local_print(f'Job to convert h5 file to DataFrame ended at : {start}, \tIt took : {dt.now() - start} s', verbose)

        return df_features, options_dataset

    except ValueError as e:
        print(e)

