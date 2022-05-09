import os
import pandas as pd
import cv2 as cv
import numpy as np
from multiprocessing.pool import ThreadPool
from plantcv import plantcv as pcv
import h5py

def update_data_dict(data_dict, key, value):
  if key not in data_dict:
    data_dict[key] = []
  data_dict[key].append(value)
  return data_dict

def is_array(x):
  '''
    Check if x is an array
  '''
  return isinstance(x, list) or isinstance(x, np.ndarray)

def store_dataset(path, dict, verbose):
  '''
    Store dataset in h5py file
    path: path of h5py file
    dict: dictionary to store
  '''
  print(F"PATH: {path}")
  h = h5py.File(path, 'w')
  
  if verbose:
    print("Saving dataset with: \n")
    

  # Saves labels
  for col in dict.keys():
    col_array = np.array(dict[col])
    shape_array = np.shape(col_array)
    if len(shape_array) == 1:
      col_array = col_array.reshape((-1, 1))
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

def bgrtogray(img):
  '''
    Convert BGR to Gray
    img: numpy array
  '''
  return cv.cvtColor(np.array(img, dtype='uint8'), cv.COLOR_BGR2GRAY)

def bgrtorgb(img):
    '''
      Convert BGR to RGB
      img: numpy array
    '''
    return cv.cvtColor(np.array(img, dtype='uint8'), cv.COLOR_BGR2RGB)


def blur_img(img, k):
    '''
      Blur image
      img: numpy array
      k: kernel size
    '''
    return cv.blur(img.copy(), k)


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


def get_gabor_img(img):
    '''
      Get gabor image
      img: numpy array
    '''

    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv.getGaborKernel(
            (ksize, ksize), 3.5, theta, 10.0, 0.5, 0, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)

    accum = np.zeros_like(img)

    def f(kern):
        return cv.filter2D(img, cv.CV_8UC3, kern)
    pool = ThreadPool(processes=8)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)

    return accum


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
