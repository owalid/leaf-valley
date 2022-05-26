import numpy as np
import cv2 as cv
from plantcv import plantcv as pcv
from matplotlib import pyplot as plt
import os
import sys

from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from utilities.utils import get_df

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
  return cv.blur(img.copy(), k)

def kmean(img, k_n):
  Z = img.reshape((-1,3))
  Z = np.float32(Z)
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
  K = k_n
  ret, label, center=cv.kmeans(Z,K,None,criteria, 50, cv.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  res = center[label.flatten()]
  return res.reshape((img.shape))

def plot_img_from_array(array_imgs, width=20, height=10):
  fig, axes = plt.subplots(1, len(array_imgs), figsize=(width, height))

  for index in range(len(array_imgs)):
    axes[index].imshow(array_imgs[index], cmap='gray')
    axes[index].axis("off")
  plt.show()

def bgr_to_rgb(img):
  return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def get_mask_bg_healthy(rgb_img):
  lab = cv.cvtColor(rgb_img, cv.COLOR_BGR2LAB)  # convert from BGR to LAB color space
  lab[:,:,0] = lab[:,:,0]/10 # L
  lab[:,:,1] += np.where(lab[:,:,1] > 125, 140, lab[:,:,1]) # A
  lab[:,:,2] = lab[:,:,2]/10 # B
  new_rgb_img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
  
  # Create mask from a channel
  a_mask = pcv.rgb2gray_lab(rgb_img=new_rgb_img, channel='a')
  a_mask = np.where(a_mask <= int(a_mask.mean()), 0, a_mask)
  a_mask = np.where(a_mask > 0, 1, a_mask)

  # Get countours and fill inside contours
  cnts = cv.findContours(a_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cv.fillPoly(a_mask, cnts, (255,255,255))
  cv.bitwise_and(rgb_img,rgb_img,mask=a_mask)

  return a_mask

def safe_open_w(path):
    '''
      Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')

def generate_img_without_bg(path_img):
    img, _, _ = pcv.readimage(path_img, mode='rgb')
    mask = get_mask_bg_healthy(img.copy())
    return mask, img

if __name__ == '__main__':
    DATA_PATH = 'data'
    SOURCE_DATASET_PATH = 'data/no_augmentation'
    DEST_DATASET_PATH = 'data/segmented_dataset'

    if not os.path.exists(DEST_DATASET_PATH):
        os.makedirs(DEST_DATASET_PATH)
        print("[+] create dataset folder for unet segmentation")

    df = get_df('./data/no_augmentation')
    indexes_species = df.index
    i = 0
    for specie in range(len(indexes_species)):
            current_df = df.loc[indexes_species[specie]]
            if current_df.specie == 'corn' and current_df.healthy:
                for index in range(1, 15):
                        im_path = f"data/no_augmentation/{indexes_species[specie]}/image ({index}).JPG"
                        ret = generate_img_without_bg(im_path)

                        if mask is not None:
                                mask, rgb_img = ret
                                mask.dtype = 'uint8'
                                cv.imwrite(f'{DEST_DATASET_PATH}/mask/{i}.png', mask*255)
                                cv.imwrite(f'{DEST_DATASET_PATH}/rgb/{i}.png', rgb_img)
                                i += 1
    print("[+] dataset generated")
