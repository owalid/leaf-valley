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

def delete_bg(specie_directory, img_number, type_img):
  path_img = f"../../data/augmentation/{specie_directory}/image ({img_number}).JPG"  

  rgb_img, _, _ = pcv.readimage(path_img, mode='rgb')
  gray_img, _, _ = pcv.readimage(path_img, mode='gray')


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


  img = rgb_img if type_img == 'color' else gray_img

  # Apply mask and add sharpness
  masked_image = pcv.apply_mask(img=img, mask=a_mask, mask_color='black')  
  im = Image.fromarray(masked_image)
  enhancer = ImageEnhance.Sharpness(im)
  im_s_1 = enhancer.enhance(2)

  if type_img == 'canny':
    edges = pcv.canny_edge_detect(np.array(im_s_1))
    return Image.fromarray(edges)

  return im_s_1

if __name__ == '__main__':
  res_augmented = input('use not augmented or augmented ?\n0 for not augmented (default)\n1 for augmented\n> ')

  if res_augmented == '1':
    res_augmented = 'no_augmented'
    df = getDf('../../data/no_augmentation')
  else:
    res_augmented = 'augmented'
    df = getDf('../../data/augmentation')

  type_img_code = input(f'Choose your result type image: \n0 for canny_edge\n1 for gray_scale (default)\n2 for rgb\n> ')
  if type_img_code == '0':
    type_img = 'canny'
  elif type_img_code == '2':
    type_img = 'color'
  else:
    type_img = 'gray'
  
  default_path_result = f"../../data/preprocess/{res_augmented}/{type_img}"
  choose_final_path = input(f'Choose your final path \n(default is: {default_path_result}/<specie>/<number>.jpg)\n> ')

  if choose_final_path == '':
    choose_final_path = default_path_result

  specie_dont_want = ['corn', 'background_without_leaves']
  df_filtred = df.loc[(~df['specie'].isin(specie_dont_want)) & (df['healthy'])]
  df_others_specie = df.loc[(~df['specie'].isin(specie_dont_want)) & (~df['specie'].isin(list(df_filtred.specie.values)))]
  df_filtred = pd.concat([df_others_specie, df_filtred])
  indexes_species = df_filtred.index

  print('\n')
  print("===========================================")
  print(f"[+] type dataset: {res_augmented}")
  print(f"[+] type image: {type_img}")
  print(f"[+] path: {choose_final_path}")
  for specie_directory in indexes_species:
    current_df = df_filtred.loc[specie_directory]
    print(f"[+] index {specie_directory}")
    print(f"[+] Start generate specie: {current_df.specie}")
    print(f"[+] Number of images: {current_df.number_img}")
    specie = current_df.specie
    
    for index in range(1, current_df.number_img):
      if int(current_df.number_img / 2) == index:
        print("[+] 50%")
      new_img = delete_bg(specie_directory, index, type_img)
      file_path = f"{choose_final_path}/{specie}/{index}.jpg"
      with safe_open_w(file_path) as f:
        new_img.save(f)
    print(f"[+] End with {specie}")