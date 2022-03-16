#!/usr/bin/python
import os
import cv2
import joblib
import numpy as np
from plantcv import plantcv as pcv
from PIL import Image, ImageEnhance
import pandas as pd
import sys
sys.path.append("/Users/agritech/vic-2_i/analysis/directory_data_repartitions")
from utils import getDf
import joblib

pcv.params.debug = ''
debug = ''

DEFAULT_FINAL_IMG_SIZE = (204,204)

HEALTHY_NOT_HEALTHY = 'HEALTHY_NOT_HEALTHY' 
ONLY_HEALTHY = 'ONLY_HEALTHY' 
NOT_HEALTHY = 'NOT_HEALTHY' 

def getDataUsed(data_used, df, type_output):
  data_used = data_used if data_used <= 1000 else -1

  if type_output == HEALTHY_NOT_HEALTHY:
    df_healthy = df.loc[df['healthy']]
    df_not_healthy = df.loc[df['healthy'] == False]
    return {
        'healthy': data_used // len(df_healthy) if data_used != -1 else -1,
        'not_healthy': data_used // len(df_not_healthy) if data_used != -1 else -1
      }

  return data_used

def get_df_filtered(df, type_output):
  df = df.loc[(df['specie'] != 'background_without_leaves')]

  if type_output == HEALTHY_NOT_HEALTHY:
    df_others_specie = df.loc[(~df['specie'].isin(list(df.specie.values)))]
    return pd.concat([df_others_specie, df])

  else:
    if type_output == ONLY_HEALTHY:
      df_filtred = df.loc[df['healthy']]
    else:
      df_filtred = df.loc[(df['healthy'] == False)]

    df_others_specie = df.loc[~df['specie'].isin(list(df_filtred.specie.values))]
    return pd.concat([df_others_specie, df_filtred])

def remove_noise(gray, num):
  Y, X = gray.shape
  nearest_neigbours = [[
      np.argmax(
          np.bincount(
              gray[max(i - num, 0):min(i + num, Y), max(j - num, 0):min(j + num, X)].ravel()))
      for j in range(X)] for i in range(Y)]
  result = np.array(nearest_neigbours, dtype=np.uint8)
  return result
  
def blur_img(img, k):
  return cv2.blur(img.copy(), k)

def kmean(img, k_n):
  Z = img.reshape((-1,3))
  Z = np.float32(Z)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
  K = k_n
  ret,label,center=cv2.kmeans(Z,K,None,criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  res = center[label.flatten()]
  return res.reshape((img.shape))

def bgr_to_rgb(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def safe_open_w(path):
    '''
      Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')

def get_mask_bg_disease(bgr_img, gray_img, specie):
  is_corn_disease = specie == 'corn'

  bgr_img_blured = blur_img(bgr_img, (3,3))
  bgr_img_blured_seven = blur_img(bgr_img, (7,7))

  # INVERSE HSV IMAGE
  hsv_inv = cv2.cvtColor(255-bgr_img_blured, cv2.COLOR_BGR2HSV)
  green_img = cv2.cvtColor(hsv_inv, cv2.COLOR_HSV2BGR)
  hsv = cv2.cvtColor(green_img, cv2.COLOR_BGR2HSV)
  hsv[:,:,1] = np.where(hsv[:,:,1] < hsv[:,:,1].mean(), hsv[:,:,1]*2, hsv[:,:,1]) # S

  # KMEAN IN HSV IMAGE
  res2 = kmean(hsv, 2)

  a_mask = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
  a_mask = np.where(a_mask <= int(a_mask.mean()), 0, a_mask)
  a_mask = np.where(a_mask != 0, 1, a_mask)

  cnts = cv2.findContours(a_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  if len(cnts) > 0:
    cv2.drawContours(a_mask, [max(cnts, key = cv2.contourArea)], -1, 255, thickness=-1)
    result = cv2.bitwise_and(bgr_img,bgr_img,mask=a_mask)

    canny = pcv.canny_edge_detect(np.array(result), sigma=0.1)

    kernel = np.ones((10,10))

    res = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    a_mask = cv2.bitwise_or(res, a_mask)

  # APPLY FIRST MASK

  # GET LAB
  lab = cv2.cvtColor(bgr_img_blured_seven, cv2.COLOR_BGR2LAB)
  lab[:,:,0] = lab[:,:,0]/10 # L
  lab[:,:,1] += np.where(lab[:,:,1] > 125, 140, lab[:,:,1]) # A
  lab[:,:,2] = lab[:,:,2]/10 # B
  lab = kmean(lab, 2)

  a_mask2 = lab.copy()
  
  a_mask2 = lab[:,:,1]
  a_mask2 = np.where(a_mask2 == a_mask2.min(), 0, a_mask2)
  a_mask2 = np.where(a_mask2 != 0, 1, a_mask2)


  if is_corn_disease:
    rgb_img_corn = bgr_to_rgb(bgr_img.copy())
    rgb_img_corn = blur_img(rgb_img_corn, (7, 7))
    
    hsv = cv2.cvtColor(rgb_img_corn, cv2.COLOR_RGB2HSV)
    hsv[:,:,1] = np.where(hsv[:,:,1] < hsv[:,:,1].mean(), hsv[:,:,1] * 2, hsv[:,:,1]) # S
    hsv = kmean(hsv, 10)

    mask_disease_orange = cv2.inRange(hsv, (5, 30, 70), (30, 255, 255))
    a_mask2 = cv2.bitwise_or(a_mask2, mask_disease_orange)
  
  # LOGICAL OR BETWEEN HSV MASK AND LAB MASK
  bitwise_or = cv2.bitwise_or(a_mask, a_mask2)
  if len(np.unique(bitwise_or)) > 1:
    # CREATE MASKED IMAGE
    masked_image_rgb = pcv.apply_mask(img=bgr_img, mask=bitwise_or, mask_color='black')
    masked_image_rgb_blur = blur_img(masked_image_rgb, (7, 7))

    hsv = cv2.cvtColor(255-masked_image_rgb_blur, cv2.COLOR_BGR2HSV)
    hsv = kmean(hsv, 2)

    final_mask = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    final_mask = np.where(final_mask <= int(final_mask.mean()), 0, final_mask)
    final_mask = np.where(final_mask != 0, 1, final_mask)

    final_mask = cv2.bitwise_or(final_mask, bitwise_or)
    return final_mask

  return bitwise_or


def get_mask_bg_healthy(rgb_img):
  lab = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
  lab[:,:,0] = lab[:,:,0]/10 # L
  lab[:,:,1] += np.where(lab[:,:,1] > 125, 140, lab[:,:,1]) # A
  lab[:,:,2] = lab[:,:,2]/10 # B
  new_rgb_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
  
  # Create mask from a channel
  a_mask = pcv.rgb2gray_lab(rgb_img=new_rgb_img, channel='a')
  a_mask = np.where(a_mask <= int(a_mask.mean()), 0, a_mask)
  a_mask = np.where(a_mask > 0, 1, a_mask)

  # Get countours and fill inside contours
  cnts = cv2.findContours(a_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cv2.fillPoly(a_mask, cnts, (255,255,255))
  cv2.bitwise_and(rgb_img,rgb_img,mask=a_mask)
  return a_mask



def generate_img_without_bg(specie_directory, img_number, type_img, specie, healthy, size_img):
  path_img = f"../data/augmentation/{specie_directory}/image ({img_number}).JPG"

  bgr_img, _, _ = pcv.readimage(path_img, mode='rgb')
  gray_img, _, _ = pcv.readimage(path_img, mode='gray')

  if healthy:
    if specie == 'corn':
      mask = np.ones_like(bgr_img)
    else:
      mask = get_mask_bg_healthy(bgr_img.copy())
  else:
    mask = get_mask_bg_disease(bgr_img.copy(), gray_img.copy(), specie)

  final_img = pcv.apply_mask(img=bgr_img, mask=mask, mask_color='black')
  final_img = bgr_to_rgb(final_img)

  im = Image.fromarray(final_img)
  enhancer = ImageEnhance.Sharpness(im)
  im_s_1 = enhancer.enhance(2)

  if type_img == 'canny':
    edges = pcv.canny_edge_detect(np.array(im_s_1))
    normalized_img = cv2.normalize(edges, edges, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return Image.fromarray(edges), cv2.resize(normalized_img, size_img)
  elif type_img == 'color+canny':
    array_img = np.array(im_s_1)
    color_normalized_img = cv2.normalize(array_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    edges = pcv.canny_edge_detect(np.array(im_s_1))
    canny_normalized_img = cv2.normalize(edges, edges, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    canny_normalized_img = cv2.resize(canny_normalized_img, size_img)
    color_normalized_img = cv2.resize(color_normalized_img, size_img)
    # final_img = np.append(color_normalized_img, canny_normalized_img)
    return color_normalized_img, canny_normalized_img
  else:
    array_img = np.array(im_s_1)
    if type_img == 'color':
      normalized_img = cv2.normalize(array_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
      normalized_img = cv2.normalize(array_img, array_img, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return im_s_1, cv2.resize(normalized_img, size_img)


# MAIN
if __name__ == '__main__':
  res_augmented = input('use not augmented or augmented ?\n0) for not augmented (default)\n1) for augmented\n> ')

  if res_augmented == '1':
    res_augmented = 'augmented'
    df = getDf('../data/augmentation')
  else:
    res_augmented = 'no_augmented'
    df = getDf('../data/no_augmentation')


  type_output = input('Type of outputs \n0) healthy_not_healthy (default)\n1) only healthy\n2) only disases\n> ')
  if type_output == '' or not type_output.isnumeric():
    type_output = HEALTHY_NOT_HEALTHY
  elif int(type_output) == 1:
    type_output = ONLY_HEALTHY
  else:
    type_output = NOT_HEALTHY

  df_filtred = get_df_filtered(df, type_output)
  indexes_species = df_filtred.index

  data_used = int(input('Choose number of images per class maximum 1000 (default maximum per class)\n> '))
  data_used = getDataUsed(data_used, df_filtred, type_output)


  type_img_code = input(f'Choose your result type image: \n0) for canny_edge\n1) for gray_scale (default)\n2) for rgb\n3) for color + canny\n> ')
  if type_img_code == '0':
    type_img = 'canny'
  elif type_img_code == '2':
    type_img = 'color'
  elif type_img_code == '3':
    type_img = 'color+canny'
  else:
    type_img = 'gray'
  


  default_path_result = f"../data/preprocess/{type_output}/{res_augmented}/{type_img}"
  choose_final_path = input(f'Choose your final path \n(default is: {default_path_result}/<specie>/<number>.jpg)\n> ')
  choose_final_path = default_path_result if choose_final_path == '' else choose_final_path
    
  generate_pickle = input(f'Want to generate pickle (default Y) ? Y/n\n> ')
  generate_pickle = 'y' if generate_pickle == '' else generate_pickle


  size_img = input(f'Size of output images (default 204px) ?\n> ')
  size_img = DEFAULT_FINAL_IMG_SIZE if size_img == '' or not size_img.isnumeric() else (int(size_img), int(size_img))



  data = dict()
  data['label'] = []
  data['img_data'] = []

  print('\n')
  print("=====================================================")
  print(f"[+] type dataset: {res_augmented}")
  print(f"[+] type output: {type_output}")
  print(f"[+] size output images: {size_img}")
  print(f"[+] type image: {type_img}")
  print(f"[+] path: {choose_final_path}")
  print("=====================================================")
  for specie_directory in indexes_species:
    current_df = df_filtred.loc[specie_directory]
    healthy = current_df.healthy
    disease = current_df.disease
    specie = current_df.specie

    current_data_used = data_used if not isinstance(data_used, dict) else data_used['healthy' if healthy else 'not_healthy']
    number_img = current_df.number_img if current_data_used == -1 else current_data_used
    print(f"[+] index {specie_directory}")
    print(f"[+] Start generate specie: {specie}")
    print(f"[+] Number of images: {number_img}")

    if type_output == HEALTHY_NOT_HEALTHY:
      label = 'healthy' if healthy else 'not_healthy'
    else:
      label = specie

    for index in range(1, number_img):
      if int(number_img / 2) == index:
        print("[+] 50%")
      clean_pill_img, np_clean_img = generate_img_without_bg(specie_directory, index, type_img, specie, healthy, size_img)
      file_path = f"{choose_final_path}/{label}/{specie}-{disease}-{index}.jpg"
      if generate_pickle.lower() == 'y':
        data['label'].append(label)
        data['img_data'].append(np_clean_img)

      with safe_open_w(file_path) as f:
        clean_pill_img.save(f)
    print(f"[+] End with {label}\n\n")
  
  if generate_pickle.lower() == 'y':
    print(f"data['img_data'].shape, {np.array(data['img_data']).shape}")
    print(f"[+] Generate pickle")
    prefix_data = 'all' if data_used == -1 else str(data_used)
    path_pickle = f"{choose_final_path}/export/data_{type_output.lower()}_{prefix_data}_{type_img}.pkl"
    os.makedirs(os.path.dirname(path_pickle), exist_ok=True)
    joblib.dump(data, path_pickle)
    print(f"[+] pickle save at {path_pickle}")