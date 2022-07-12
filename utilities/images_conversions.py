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
