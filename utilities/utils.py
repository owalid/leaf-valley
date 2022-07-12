import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import numpy as np
from multiprocessing.pool import ThreadPool

from plantcv import plantcv as pcv
import h5py
import json

import os.path as path
import sys
from inspect import getsourcefile
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
current_dir = current_dir[:current_dir.rfind(path.sep)]
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from utilities.prepare_features import prepare_features
from utilities.remove_background_functions import remove_bg
from utilities.images_conversions import rgbtobgr

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

def preprocess_prediction(img, options):
  '''
    Preprocess image before prediction
  '''
  mask, masked_img = (None, None)
  
  if 'normalize_type' in options.keys() and options['normalize_type'] and isinstance(options['normalize_type'], str):
    img = cv.normalize(img, None, alpha=0, beta=1, norm_type=CV_NORMALIZE_TYPE[options['normalize_type']], dtype=cv.CV_32F)

  if 'size_img' in options.keys() and options['size_img'] is not None and isinstance(options['size_img'], tuple):
    img = cv.resize(img, options['size_img'])

  if 'crop_img' in options.keys() and options['crop_img']:
    img = crop_resize_image(img, img)
  
  if 'should_remove_bg' in options.keys() and options['should_remove_bg']:
    bgr_img = rgbtobgr(img)
    mask, masked_img = remove_bg(bgr_img)
    
  if ('features' in options.keys() and options['features'] is not None) or (options['features'] and len(options['features']) == 1 and options['features'][0] != 'rgb'):
    data = {}
    is_deep_learning_feature = ['rgb', 'lab', 'hsv', 'canny', 'gray', 'gabor'] in options['features'] and len(options['features']) == 1
    img = prepare_features(data, img, options['features'], masked_img=masked_img, mask=mask, is_deep_learning_features=is_deep_learning_feature)
    
  return img
  

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


def crop_resize_image(img_masked, img_to_resize):
    '''
      Crop and resize image of leaf to delete padding arround leaf.

      img_masked: numpy array
      img_to_resize: numpy array
      return numpy array
    '''
    arr = 1*(img_masked.sum(axis=1) > 0)
    x1 = list(arr).index(1)
    x2 = len(arr) - list(arr)[::-1].index(1)
    arr = 1*(img_masked.sum(axis=0) > 0)
    y1 = list(arr).index(1)
    y2 = len(arr) - list(arr)[::-1].index(1)

    return cv.resize(img_to_resize[x1:x2+1, y1:y2+1, ], img_masked.shape, interpolation=cv.INTER_CUBIC)

def safe_open_w(path):
    '''
      Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')


def get_canny_img(img, sigma=1.5):
    '''
      Get canny image
      img: numpy array
      sigma: sigma of gaussian
      return numpy array
    '''

    return pcv.canny_edge_detect(img, sigma=sigma)


def kmean_img(img, k_n):
    '''
      K-mean clustering image

      img: numpy array
      k_n: number of clusters
      return: numpy array
    '''
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
    K = k_n
    ret, label, center = cv.kmeans(
        Z, K, None, criteria, 50, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))


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
