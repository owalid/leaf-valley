import sys
import os.path as path
import numpy as np

import cv2 as cv
import plantcv as pcv
from inspect import getsourcefile
from PIL import Image, ImageEnhance

current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from utilities.image_transformation import bgrtogray, crop_resize_image
from utilities.remove_background_functions import remove_bg
from utilities.extract_features import get_pyfeats_features, get_lpb_histogram, get_hue_moment, get_haralick, get_hsv_histogram, get_lab_histogram, get_graycoprops, get_lab_img, get_hsv_img, get_gabor_img, get_canny_img


def update_features_dict(data_dict, key, value):
  if key not in data_dict:
    data_dict[key] = []
  data_dict[key].append(value)
  return data_dict

def prepare_features(data, rgb_img, target_features, should_remove_bg, size_img=None, normalize_type=False, crop_img=False, type_img='rgb', is_deep_learning_features=False):
  '''
    Preprocess image before prediction and trainning.
    
    Parameters:
      - data: dictionary to store features (type: dictionary)
      - rgb_img: image to apply features (type: numpy.ndarray)
      - target_features: list of features to apply (type: list)
      - should_remove_bg: should remove background (type: boolean)
      - size_img: size of image (type: tuple)
      - normalize_type: normalization type from opencv (type: int)
      - crop_img: should crop image (type: boolean)
      - type_img: type of image (type: string)
      - is_deep_learning_features: should apply deep learning features (type: boolean)
      
    Returns:
      - data: dictionary with features (type: dictionary)
      - pil_img: image with preprocessing for saving in jpg (type: PIL.Image)
  '''
  
  
  # ==== Preprocess image ====
  
  # Crop image
  if crop_img == True:
      rgb_img = crop_resize_image(rgb_img, rgb_img)
  
  # Generate PIL image and add enhancements
  im = Image.fromarray(rgb_img)
  enhancer = ImageEnhance.Sharpness(im)
  pill_img = enhancer.enhance(2)
  rgb_img = np.array(pill_img)
  
  # Remove bg
  mask, rgb_img = remove_bg(rgb_img) if should_remove_bg else (None, rgb_img)
  
  if size_img is not None and isinstance(size_img, tuple):
    rgb_img = cv.resize(rgb_img, size_img)

  # Transform image to type_img
  if type_img == 'canny':
      edges = pcv.canny_edge_detect(rgb_img)
      pill_img = Image.fromarray(edges)
  elif type_img == 'gray':
      gray_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
      pill_img = Image.fromarray(gray_img)
  elif type_img == 'gabor':
      gabor_img = get_gabor_img(rgb_img)
      pill_img = Image.fromarray(gabor_img)
      
  if normalize_type:
    rgb_img = cv.normalize(rgb_img, None, alpha=0, beta=1, norm_type=normalize_type, dtype=cv.CV_32F)
  
  
  # ==== Extract feature ====
  
  # FEATURES DEEP LEARNING
  if 'rgb' in target_features:
      data = update_features_dict(data, 'rgb_img', rgb_img)
  if 'gabor' in target_features:
      if is_deep_learning_features:
        return get_gabor_img(rgb_img)
      
      data = update_features_dict(
          data, 'gabor_img', get_gabor_img(rgb_img))
  if 'gray' in target_features:
      if is_deep_learning_features:
        return bgrtogray(rgb_img)
      
      data = update_features_dict(data, 'gray_img', bgrtogray(rgb_img))
  if 'canny' in target_features:
      if is_deep_learning_features:
        return get_canny_img(rgb_img)
      
      data = update_features_dict(
          data, 'canny_img', get_canny_img(rgb_img))
  if 'lab' in target_features:
      if is_deep_learning_features:
        return get_lab_img(rgb_img)
      
      data = update_features_dict(data, 'lab',  get_lab_img(rgb_img))
  if 'hsv' in target_features:
      if is_deep_learning_features:
        return get_hsv_img(rgb_img)
      
      data = update_features_dict(data, 'hsv',  get_hsv_img(rgb_img))

  # FEATURES MACHINE LEARNING
  if 'graycoprops' in target_features:
      features = get_graycoprops(rgb_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'lpb_histogram' in target_features:
      features = get_lpb_histogram(rgb_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'hue_moment' in target_features:
      features = get_hue_moment(rgb_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'haralick' in target_features:
      features = get_haralick(rgb_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'histogram_hsv' in target_features:
      features = get_hsv_histogram(rgb_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'histogram_lab' in target_features:
      features = get_lab_histogram(rgb_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'pyfeats' in target_features and rgb_img:
      if mask is None:
        mask, rgb_img = remove_bg(rgb_img)
        
      pyfeats_features = get_pyfeats_features(rgb_img, mask)
      for feature in pyfeats_features:
          data = update_features_dict(
              data, feature, pyfeats_features[feature])
          
  return data, pill_img
