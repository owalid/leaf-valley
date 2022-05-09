'''
  CLI used to preprocess the data and get features and classes.
'''

import random
import os
from tabnanny import verbose
import cv2 as cv
import joblib
import numpy as np
from plantcv import plantcv as pcv
from PIL import Image, ImageEnhance
import pandas as pd
import argparse as ap
from argparse import RawTextHelpFormatter

from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from utilities.utils import crop_resize_image, safe_open_w, get_df, get_canny_img, get_gabor_img, store_dataset, update_data_dict, bgrtogray
from utilities.extract_features import get_pyfeats_features, get_lbp_histogram, get_hue_moment, get_haralick, get_hsv_histogram, get_lab_histogram, get_graycoprops
from utilities.remove_background_functions import remove_bg

pcv.params.debug = ''
debug = ''

VERBOSE = False
DEFAULT_FINAL_IMG_SIZE = (256, 256)

HEALTHY_NOT_HEALTHY = 'HEALTHY_NOT_HEALTHY'
ONLY_HEALTHY = 'ONLY_HEALTHY'
NOT_HEALTHY = 'NOT_HEALTHY'
ALL = 'ALL'

CV_NORMALIZE_TYPE = {
    'NORM_INF': cv.NORM_INF,
    'NORM_L1': cv.NORM_L1,
    'NORM_L2': cv.NORM_L2,
    'NORM_L2SQR': cv.NORM_L2SQR,
    'NORM_HAMMING': cv.NORM_HAMMING,
    'NORM_HAMMING2': cv.NORM_HAMMING2,
    'NORM_TYPE_MASK': cv.NORM_TYPE_MASK,
    'NORM_RELATIVE': cv.NORM_RELATIVE,
    'NORM_MINMAX': cv.NORM_MINMAX
}

def local_print(msg):
    if VERBOSE:
        print(msg)
def get_data_used(data_used, df, type_output):

    if type_output == HEALTHY_NOT_HEALTHY:
        df_healthy = df.loc[df['healthy']]
        df_not_healthy = df.loc[df['healthy'] == False]
        return {
            'healthy': data_used // len(df_healthy) if data_used != -1 else -1,
            'not_healthy': data_used // len(df_not_healthy) if data_used != -1 else -1
        }

    data_used = data_used if data_used <= 1000 else -1
    return data_used


def get_df_filtered(df, type_output):
    df = df.loc[(df['specie'] != 'background_without_leaves')]

    if type_output == ALL:
        return df
    if type_output == HEALTHY_NOT_HEALTHY:
        df_others_specie = df.loc[(~df['specie'].isin(list(df.specie.values)))]
        return pd.concat([df_others_specie, df])
    else:
        if type_output == ONLY_HEALTHY:
            df_filtred = df.loc[df['healthy']]
        else:
            df_filtred = df.loc[(df['healthy'] == False)]

        df_others_specie = df.loc[~df['specie'].isin(
            list(df_filtred.specie.values))]
        return pd.concat([df_others_specie, df_filtred])


def generate_img_without_bg(specie_directory, img_number, type_img, size_img, cropped_img, normalize_img, normalized_type):
    path_img = f"../data/augmentation/{specie_directory}/image ({img_number}).JPG"
    bgr_img, _, _ = pcv.readimage(path_img, mode='bgr')
    mask, new_img = remove_bg(bgr_img)

    if cropped_img == True:
        new_img = crop_resize_image(new_img, new_img)

    im = Image.fromarray(new_img)
    enhancer = ImageEnhance.Sharpness(im)
    pill_img = enhancer.enhance(2)
    array_img = np.array(pill_img)
    array_img = cv.resize(array_img, size_img)
    
    if type_img == 'canny':
        edges = pcv.canny_edge_detect(array_img)
        pill_img = Image.fromarray(edges)
    elif type_img == 'gray':
        gray_img = cv.cvtColor(array_img, cv.COLOR_BGR2GRAY)
        pill_img = Image.fromarray(gray_img)
    elif type_img == 'gabor':
        gabor_img = get_gabor_img(array_img)
        pill_img = Image.fromarray(gabor_img)
    if normalize_img:
        array_img = cv.normalize(array_img, None, alpha=0, beta=1, norm_type=normalized_type, dtype=cv.CV_32F)

    return pill_img, array_img, bgr_img, mask


# MAIN
if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-a", "--augmented", required=False, action='store_true', default=False, help='Use directory augmented')
    parser.add_argument("-src", "--src-directory", required=False, type=str, default='', help='Directory source who can find images. default (data/{augmented})')
    parser.add_argument("-wi", "--write-img", required=False, action='store_true', default=False, help='Write images (png) in the new directory')
    parser.add_argument("-crop", "--crop-img", required=False, action='store_true', default=False, help='Remove padding around leaf')
    parser.add_argument("-nor", "--normalize-img", required=False, action='store_true', default=True, help='Normalize images, you can specify the normalization type with the option -nortype')
    parser.add_argument("-nortype", "--normalize-type", required=False, type=str, default='NORM_MINMAX', help='Normalize images features with cv.normalize (Default: NORM_MINMAX) \nTypes: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_NormTypes.html')
    parser.add_argument("-c", "--classification", required=False, type=str, default="HEALTHY_NOT_HEALTHY", help='Classification type: HEALTHY_NOT_HEALTHY(default), ONLY_HEALTHY, NOT_HEALTHY, ALL')
    parser.add_argument("-n", "--number-img", required=False, type=int, default=1000, help='Number of images to use per class to select maximum of all classes use -1. (default 1000)')
    parser.add_argument("-rt", "--result-type", required=False, type=str, default="GRAY", help='Type of result image for DP: GRAY, GABOR, CANNY, RGB. (default: GRAY)')
    parser.add_argument("-dst", "--destination", required=False, type=str, default='', help='Path to save the data. (default: data/preprocess)')
    parser.add_argument("-f", "--features", required=False, type=str, help='Features to extract separate by ","\nExample: -f=graycoprops,lpb_histogram,hue_moment\nList of features:\n   - For DP: rgb, gray, canny, gabor\n   - For ML: graycoprops, lpb_histogram, hue_moment, haralick, histogram_hsv, histogram_lab, pyfeats')
    parser.add_argument("-s", "--size", required=False, type=int, default=256, help='Size of images. (default 256x256)')
    parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help='Verbose')
    args = parser.parse_args()
    print(args)
    
    random.seed(42)
    normalize_type = args.normalize_type
    normalize_type = 'NORM_MINMAX' if normalize_type not in CV_NORMALIZE_TYPE.keys() else normalize_type
    
    res_augmented = 'augmentation' if args.augmented else 'no_augmentation'
    src_directory = os.path.join(args.src_directory, res_augmented) if args.src_directory != '' else f"data/{res_augmented}"
    df = get_df(src_directory)
    type_output = args.classification
    write_img = args.write_img
    crop_img = args.crop_img
    normalize_img = args.normalize_img
    df_filtred = get_df_filtered(df, type_output)
    indexes_species = df_filtred.index
    if len(indexes_species) == 0:
        print("No images to process")
        exit()

    data_used = args.number_img
    type_img = args.result_type.lower()
    
    if args.destination == '' and args.src_directory != '':
        dest_path = f"{args.src_directory}/preprocess/{type_output}/{res_augmented}"
    elif args.destination != '':
        dest_path = args.destination
    else:
        dest_path = f'data/preprocess/{type_output}/{res_augmented}'

    if not os.path.exists(dest_path): # Create a dest_path if not exist. 
        os.makedirs(dest_path)
        print("The new directory is created!")
        
    answers_type_features = args.features.replace(' ', '').split(',') if args.features != None else []
    answers_type_features = ['graycoprops', 'lpb_histogram', 'hue_moment', 'haralick', 'histogram_hsv', 'histogram_lab', 'pyfeats'] if len(answers_type_features) == 0 else answers_type_features
    
    size_img = (args.size, args.size) if args.size > 0 else DEFAULT_FINAL_IMG_SIZE
    VERBOSE = args.verbose
    series = []
    local_print('\n')
    local_print("=====================================================")
    local_print(f"[+] type dataset: {res_augmented}")
    local_print(f"[+] type output: {type_output}")
    local_print(f"[+] size output images: {size_img}")
    local_print(f"[+] type image: {type_img}")
    local_print(f"[+] path: {dest_path}")
    if normalize_img:
        local_print(f"[+] Normalized type: cv.{normalize_type}")
        
    if len(answers_type_features) > 0:
        local_print(f"[+] answers_type_features: {answers_type_features}")
    data = dict()
    local_print("=====================================================")
    for specie_directory in indexes_species:
        current_df = df_filtred.loc[specie_directory]
        healthy = current_df.healthy
        disease = current_df.disease
        specie = current_df.specie

        current_data_used = data_used if not isinstance(data_used, dict) else data_used['healthy' if healthy else 'not_healthy']

        if type_output == HEALTHY_NOT_HEALTHY:
            label = 'healthy' if healthy else 'not_healthy'
        elif type_output == NOT_HEALTHY:
            label = disease
        elif type_output == ALL:
            label = f"{specie}_{disease}"
        else:
            label = specie
        
        if current_data_used == -1 or current_df.number_img <= current_data_used:
            number_img = current_df.number_img
            indexes = list(range(1, current_df.number_img+1))
        else:
            number_img = current_data_used
            indexes = random.sample(list(range(1, current_df.number_img)), number_img) # Get alls indexes with random without repetition.
        local_print(f"[+] index {specie_directory}")
        local_print(f"[+] Start generate specie: {specie}")
        local_print(f"[+] Number of images: {number_img}")
        
        for index in indexes:
            if len(indexes) // 2 == np.where(indexes == index):
                local_print("[+] 50%")
            pill_masked_img, masked_img, raw_img, mask = generate_img_without_bg(specie_directory, index, type_img, size_img, crop_img, normalize_img, CV_NORMALIZE_TYPE[normalize_type])
            file_path = f"{dest_path}/{label}/{specie}-{disease}-{index}.jpg"
            specie_index = f"{specie}_{disease}_{index}"
            data = update_data_dict(data, 'labels', specie_index)
            data = update_data_dict(data, 'classes', disease)
            
            # FEATURES DEEP LEARNING
            if 'rgb' in answers_type_features or len(answers_type_features) == 0:
                data = update_data_dict(data, 'rgb_img',  masked_img)
            if 'gabor' in answers_type_features:
                data = update_data_dict(data, 'gabor_img',  get_gabor_img(masked_img))
            if 'gray' in answers_type_features:
                data = update_data_dict(data, 'gray_img',  bgrtogray(masked_img))
            if 'canny' in answers_type_features:
                data = update_data_dict(data, 'canny_img',  get_canny_img(masked_img))
                
            # FEATURES MACHINE LEARNING
            if 'graycoprops' in answers_type_features:
                data = update_data_dict(data, 'graycoprops',  get_graycoprops(masked_img))
            if 'lpb_histogram' in answers_type_features:
                data = update_data_dict(data, 'lpb_histogram',  get_lbp_histogram(masked_img))
            if 'hue_moment' in answers_type_features:
                data = update_data_dict(data, 'hue_moment',  get_hue_moment(masked_img))
            if 'haralick' in answers_type_features:
                data = update_data_dict(data, 'haralick',  get_haralick(masked_img))
            if 'histogram_hsv' in answers_type_features:
                data = update_data_dict(data, 'histogram_hsv',  get_hsv_histogram(masked_img))
            if 'histogram_lab' in answers_type_features:
                data = update_data_dict(data, 'histogram_lab',  get_lab_histogram(masked_img))
            if 'pyfeats' in answers_type_features:
                pyfeats_features = get_pyfeats_features(raw_img, mask)
                for feature in pyfeats_features:
                    data = update_data_dict(data, feature, pyfeats_features[feature])
                    
            if write_img:
                with safe_open_w(file_path) as f:
                    pill_masked_img.save(f)
        local_print(f"[+] End with {label}\n\n")
    
    local_print(f"Number of images, {len(data)}")
    local_print(f"[+] Generate hdf5 file")
    prefix_data = 'all' if int(data_used) == -1 else str(data_used - 1)
    path_hdf = f"{dest_path}/export/data_{type_output.lower()}_{prefix_data}_{type_img}.h5"
    os.makedirs(os.path.dirname(path_hdf), exist_ok=True)
    store_dataset(path_hdf, data, VERBOSE)
    local_print(f"[+] pickle save at {path_hdf}")
