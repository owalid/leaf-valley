import cv2 as cv
import numpy as np

def bgrtogray(img):
  '''
    Convert BGR to Gray
    img: numpy array
  '''
  return cv.cvtColor(np.array(img, dtype='uint8'), cv.COLOR_BGR2GRAY)

def rgbtobgr(img):
    '''
      Convert RGB to BGR
      img: numpy array
    '''
    return cv.cvtColor(np.array(img, dtype='uint8'), cv.COLOR_BGR2RGB)
  
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