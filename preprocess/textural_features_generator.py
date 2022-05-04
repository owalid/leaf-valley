#####
#####   Textural Features from images generator
#####

import os
import time
import random as rd
import argparse as ap
import pandas as pd 
import concurrent.futures
from itertools import repeat
import textural_features as tf


def process_img(path_name, path_img):
    df = pd.DataFrame()
    return tf.textural_features_generator(df, path_name, path_img)  

def main(path_name):
    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_img, repeat(path_name), filenames)

    df_features = pd.DataFrame()

    for df in results:
        df_features = pd.concat([df_features, df])

    print(df_features.shape)
    df_features.to_parquet(os.path.join(path_name,'df_texturalfeaturesofimages.parquet'))

    print(f'Finished in {time.perf_counter()-start:.2f} second(s)')

if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("-p", "--data_path_name", required=True, type=str,help='The path of the data')
    parser.add_argument("-n", "--img_number", required=True, type=int,help='Number of images by plant class (-1 to process all images)')
    args = parser.parse_args()
    path_name = args.data_path_name

    N = args.img_number 

    filenames = []
    for d in [d for d in os.listdir(path_name) if os.path.isdir(os.path.join(path_name,d))]:
        img_lst = []
        for f in os.listdir(os.path.join(path_name,d)):
            img_lst.append(os.path.join(d,f))
        filenames.append(rd.sample(img_lst, len(img_lst) if (N==-1)|(N>len(img_lst)) else N))

    filenames = sum(filenames,[])

    print(len(filenames))

    main(path_name)
