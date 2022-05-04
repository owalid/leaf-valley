import os
import pandas as pd
import cv2 as cv
import numpy as np
from multiprocessing.pool import ThreadPool
from plantcv import plantcv as pcv


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


def get_df(path='../../data/augmentation'):
    '''
      Get dataframe from path of datasets
      path: path of datasets

      return pandas dataframe
    '''
    all_folder = os.listdir(path)
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
