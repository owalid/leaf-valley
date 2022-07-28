import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import numpy as np
from multiprocessing.pool import ThreadPool

from plantcv import plantcv as pcv
import h5py
import json

# import os.path as path
# import sys
# from inspect import getsourcefile
# current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
# current_dir = current_dir[:current_dir.rfind(path.sep)]
# sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
# from utilities.prepare_features import prepare_features
# from utilities.remove_background_functions import remove_bg
# from utilities.image_transformation import rgbtobgr

CV_NORMALIZE_TYPE = {
    'NORM_INF': cv.NORM_INF,
    'NORM_L1': cv.NORM_L1,
    'NORM_L2': cv.NORM_L2,
    'NORM_L2SQR': cv.NORM_L2SQR,
    'NORM_HAMMING': cv.NORM_HAMMING,
    'NORM_HAMMING2': cv.NORM_HAMMING2,
    'NORM_TYPE_MASK': cv.NORM_TYPE_MASK,
    'NORM_RELATIVE': cv.NORM_RELATIVE,
    'NORM_MINMAX': cv.NORM_MINMAX
}

  
def update_data_dict(data_dict, key, value):
  if key not in data_dict:
    data_dict[key] = []
  data_dict[key].append(value)
  return data_dict

def safe_get_item(dictionary, key, default=None):
    '''
      Get item from dictionary
      dictionary: dictionary
    '''
    return dictionary[key] if key in dictionary else default

# def preprocess_pipeline_prediction(rgb_img, options):
#   '''
#     Preprocess image before prediction
#   '''
  
#   normalize_type = None
#   if 'normalize_type' in options.keys() and options['normalize_type'] and isinstance(options['normalize_type'], str) and options['normalize_type'] in CV_NORMALIZE_TYPE.keys():
#     normalize_type = CV_NORMALIZE_TYPE[options['normalize_type']]

#   norm_type = safe_get_item(options, 'normalize_type', None)
#   norm_type = CV_NORMALIZE_TYPE[norm_type] if norm_type is not None else None
#   data = {}
#   img = prepare_features(data, rgb_img, safe_get_item(options,'features',{}), safe_get_item(options, 'should_remove_bg'),
#                         size_img=safe_get_item(options, 'size_img', None),\
#                         normalize_type=normalize_type,\
#                         crop_img=safe_get_item(options, 'crop_img', False),\
#                         is_deep_learning_features=safe_get_item(options, 'crop_img', False))
    
#   return img
  

def chunks(arr, chunk_size):
  '''
    Split array into chunks
    Args:
      arr: array to split
      chunk_size: size of chunks
    Returns:
      list of chunks
  '''
  return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]


def is_array(x):
  '''
    Check if x is an array
  '''
  return isinstance(x, list) or isinstance(x, np.ndarray)

def get_dataset(path):
  '''
    Get dataset from h5py file
  '''
  print(F"PATH: {path}")
  hf = h5py.File(path, 'r')
  return hf
  
def store_dataset(path, src_dict, verbose):
  '''
    Store dataset in h5py file
    path: path of h5py file
    src_dict: dictionary to store
  '''
  print(F"PATH: {path}")
  h = h5py.File(path, 'w')
  
  if verbose:
    print("Saving dataset with: \n")
    

  # Saves labels
  for col in src_dict.keys():
    if isinstance(src_dict[col], dict):
      str_json = json.dumps(src_dict[col])
      h.create_dataset(col, (1,), h5py.string_dtype('utf-8'), data=[str_json])
    else:
      col_array = np.array(src_dict[col])
      shape_array = np.shape(col_array)
      
      
      first_element = col_array[0]
      
      while is_array(first_element):  # If array, keep going
        first_element = first_element[0]

      # Select the correct type for h5py file
      if type(first_element) is float or type(first_element) is np.float64 or type(first_element) is np.float32: # If float
        col_type = h5py.h5t.IEEE_F32BE
        col_type_str = "h5py.h5t.IEEE_F32BE"
      elif type(first_element) is bool or type(first_element) is np.uint8:
        col_array.astype(np.uint8)
        col_type = h5py.h5t.STD_U8BE
        col_type_str = "h5py.h5t.STD_U8BE"
      elif type(first_element) is int or type(first_element) is np.int64 or type(first_element) is np.int32: # If int or int64
        col_type = h5py.h5t.STD_I32BE
        col_type_str = "h5py.h5t.STD_I32BE"
      elif type(first_element) is np.str_ or type(first_element) is str: # If string or np.str
        col_type = h5py.string_dtype('utf-8')
        col_type_str = "h5py.string_dtype('utf-8')"
        
      if verbose:
        print(f"[+] Column: {col} - Type: {col_type_str} - Shape: {shape_array}")
      
      col_array = np.array(col_array, dtype=col_type)
        
      # Create the dataset
      h.create_dataset(col, shape_array, col_type, data=col_array)

def replace_text(text, lst, rep=' '):
    '''
      Replace text in list with rep
    '''
    for l in lst:
        text = text.replace(l, rep)
    return text

def safe_open_w(path, option_open='w'):
    '''
      Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, option_open)


def get_df(path='data/augmentation'):
    '''
      Get dataframe from path of datasets
      path: path of datasets

      return pandas dataframe
    '''
    all_folder = [d for d in os.listdir(path) if '__' in d]
    df = pd.DataFrame(columns=['number_img', 'disease',
                      'disease_family', 'healthy', 'specie'], index=all_folder)

    for name_folder in all_folder:
        files = os.listdir(f"{path}/{name_folder}")
        name_splited = name_folder.split('___')
        df.loc[name_folder].specie = name_splited[0].lower()
        df.loc[name_folder].number_img = len(files)
        df.loc[name_folder].disease = name_splited[-1].lower()
        df.loc[name_folder].disease_family = df.loc[name_folder].disease.split(
            '_')[-1].replace(')', '')
        df.loc[name_folder].healthy = name_splited[-1] == 'healthy'
    return df

def set_plants_dict(df):
    d = {}
    for specie in ['All']+sorted(df.specie.unique()):
        d[specie] = {}
        d[specie] = list(sorted(df.loc[((df.specie==specie)|(specie=='All'))].disease.unique()))
        if len(d[specie])>1:
            d[specie] = ['All']+d[specie]

    return d

