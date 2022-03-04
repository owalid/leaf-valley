#!/usr/bin/python
import os
import cv2
import numpy as np
from plantcv import plantcv as pcv
from PIL import Image, ImageEnhance
import pandas as pd
import sys
sys.path.append("/Users/agritech/vic-2_i/analysis/directory_data_repartitions")
from utils import getDf

pcv.params.debug = ''
debug = ''

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')

def delete_bg(specie, img_number, gray_scale):
  path_img = f"../../data/augmentation/{specie.capitalize()}___healthy/image ({img_number}).JPG"  


  rgb_img, path, filename = pcv.readimage(path_img, mode='rgb')
  gray_img, path_grat, filename_gray = pcv.readimage(path_img, mode='gray')


  lab = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
  lab[:,:,0] = lab[:,:,0]/10 # L
  lab[:,:,1] += np.where(lab[:,:,1] > 125, 140, lab[:,:,1]) # A
  lab[:,:,2] = lab[:,:,2]/10 # B
  new_rgb_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
  
  a_mask = pcv.rgb2gray_lab(rgb_img=new_rgb_img, channel='a')
  a_mask = np.where(a_mask <= int(a_mask.mean()), 0, a_mask)
  a_mask = np.where(a_mask > 0, 1, a_mask)
  cnts = cv2.findContours(a_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cv2.fillPoly(a_mask, cnts, (255,255,255))
  result = cv2.bitwise_and(rgb_img,rgb_img,mask=a_mask)

  masked_image = pcv.apply_mask(img=gray_img, mask=a_mask, mask_color='black')

  im = Image.fromarray(masked_image)
  enhancer = ImageEnhance.Sharpness(im)
  factor = 2
  im_s_1 = enhancer.enhance(factor)

  if gray_scale:
    return im_s_1


  edges = pcv.canny_edge_detect(np.array(im_s_1))
  return Image.fromarray(edges)




if __name__ == '__main__':
  res_augmented = input('use not augmented or augmented ?\n0 for not augmented\n1 for augmented\n> ')
  res_augmented = int(res_augmented)

  if res_augmented not in [0, 1]:
    print('you need to select 0 or 1')
    exit(1)
  if res_augmented == 0:
    res_augmented = 'no_augmented'
    df = getDf('../../data/no_augmentation')
  else:
    res_augmented = 'augmented'
    df = getDf('../../data/augmentation')

  gray_scale = input(f'Choose your result image: \n0 for canny_edge\n1 for gray_scale\n> ')
  gray_scale = bool(int(gray_scale))
  
  if gray_scale:
    gray_scale_prefix = 'gray'
  else:
    gray_scale_prefix = 'canny'
  
  default_path_result = f"../../data/preprocess/{res_augmented}/{gray_scale_prefix}"
  choose_final_path = input(f'choose your final path \n(default is: {default_path_result}/<specie>/<number>.jpg)\n> ')

  if choose_final_path == '':
    choose_final_path = default_path_result


  df_filtred = df.loc[(df['specie'] != 'corn') & (df['healthy'])]
  indexes_species = df_filtred.index

  for index_specie in indexes_species:
    current_df = df_filtred.loc[index_specie]
    print(f"[+] specie: {current_df.specie}")
    print(f"[+] number of images: {current_df.number_img}")
    specie = current_df.specie
    for index in range(1, current_df.number_img):
      if int(current_df.number_img / 2) == index:
        print("[+] 50%")
      new_img = delete_bg(specie, index, gray_scale)
      file_path = f"{choose_final_path}/{specie}/{index}.jpg"
      with safe_open_w(file_path) as f:
        new_img.save(f)
    print(f"---- end with {specie} ----")
  