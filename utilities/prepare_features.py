import os.path as path
import sys
from inspect import getsourcefile
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from utilities.images_conversions import bgrtogray, rgbtobgr
from utilities.remove_background_functions import remove_bg
from utilities.extract_features import get_pyfeats_features, get_lpb_histogram, get_hue_moment, get_haralick, get_hsv_histogram, get_lab_histogram, get_graycoprops, get_lab_img, get_hsv_img, get_gabor_img


def update_features_dict(data_dict, key, value):
  if key not in data_dict:
    data_dict[key] = []
  data_dict[key].append(value)
  return data_dict

def prepare_features(data, img, features, masked_img=None, mask=None, is_deep_learning_features=False):
  '''
    Apply features to image
  '''
  
  masked_img = img if masked_img is None else masked_img
  
  if 'rgb' in features:
      data = update_features_dict(data, 'rgb_img', masked_img)
  if 'gabor' in features:
      if is_deep_learning_features:
        return get_gabor_img(masked_img)
      
      data = update_features_dict(
          data, 'gabor_img', get_gabor_img(masked_img))
  if 'gray' in features:
      if is_deep_learning_features:
        return bgrtogray(masked_img)
      
      data = update_features_dict(data, 'gray_img', bgrtogray(masked_img))
  if 'canny' in features:
      if is_deep_learning_features:
        return get_canny_img(masked_img)
      
      data = update_features_dict(
          data, 'canny_img', get_canny_img(masked_img))
  if 'lab' in features:
      if is_deep_learning_features:
        return get_lab_img(masked_img)
      
      data = update_features_dict(data, 'lab',  get_lab_img(masked_img))
  if 'hsv' in features:
      if is_deep_learning_features:
        return get_hsv_img(masked_img)
      
      data = update_features_dict(data, 'hsv',  get_hsv_img(masked_img))

  # FEATURES MACHINE LEARNING
  if 'graycoprops' in features:
      features = get_graycoprops(masked_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'lpb_histogram' in features:
      features = get_lpb_histogram(masked_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'hue_moment' in features:
      features = get_hue_moment(masked_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'haralick' in features:
      features = get_haralick(masked_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'histogram_hsv' in features:
      features = get_hsv_histogram(masked_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'histogram_lab' in features:
      features = get_lab_histogram(masked_img)
      for feature in features:
          data = update_features_dict(data, feature, features[feature])
  if 'pyfeats' in features:
      if mask is None or masked_img is None:
        bgr_img = rgbtobgr(img)
        mask, masked_img = remove_bg(bgr_img)
        
      pyfeats_features = get_pyfeats_features(img, mask)
      for feature in pyfeats_features:
          data = update_features_dict(
              data, feature, pyfeats_features[feature])
  return data
