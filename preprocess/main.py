'''
  CLI used to preprocess the data and get features and classes.
'''
import concurrent.futures
from itertools import repeat
import multiprocessing as mp
from tqdm import tqdm
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
from utilities.extract_features import get_pyfeats_features, get_lpb_histogram, get_hue_moment, get_haralick, get_hsv_histogram, get_lab_histogram, get_graycoprops, get_lab_img, get_hsv_img
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

def generate_img(path_img, type_img, size_img, cropped_img, normalize_img, normalized_type):
    bgr_img, _, _ = pcv.readimage(path_img, mode='bgr')
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)

    if cropped_img == True:
        rgb_img = crop_resize_image(rgb_img, rgb_img)
    im = Image.fromarray(rgb_img)
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

    return pill_img, array_img, bgr_img


def generate_img_without_bg(path_img, type_img, size_img, cropped_img, normalize_img, normalized_type):
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

def multiprocess_worker(specie_directory, df_filtred, data_used, type_output, src_directory, dest_path, size_img, crop_img, normalize_img, normalize_type, type_img, should_remove_bg, answers_type_features, write_img):
    current_df = df_filtred.loc[specie_directory]
    healthy = current_df.healthy
    disease = current_df.disease
    specie = current_df.specie
    data = dict()
    current_data_used = data_used if not isinstance(data_used, dict) else data_used['healthy' if healthy else 'not_healthy']

    if type_output == HEALTHY_NOT_HEALTHY:
        class_name = 'healthy' if healthy else 'not_healthy'
    elif type_output == NOT_HEALTHY:
        class_name = disease
    elif type_output == ALL:
        class_name = f"{specie}_{disease}"
    else:
        class_name = specie
    
    if current_data_used == -1 or current_df.number_img <= current_data_used:
        number_img = current_df.number_img
        indexes = list(range(1, current_df.number_img+1))
    else:
        number_img = current_data_used
        indexes = random.sample(list(range(1, current_df.number_img)), number_img) # Get alls indexes with random without repetition.
        
    local_print(f"\n[+] index {specie_directory}")
    local_print(f"[+] Start generate specie: {specie}")
    local_print(f"[+] Number of images: {number_img}")

    for index in tqdm(indexes, ncols=100) if VERBOSE else indexes:
        path_img = os.path.join(src_directory, specie_directory, f"image ({index}).JPG")
        
        if should_remove_bg:
            pill_masked_img, masked_img, raw_img, mask = generate_img_without_bg(path_img, type_img, size_img, crop_img, normalize_img, CV_NORMALIZE_TYPE[normalize_type])
        else:
            pill_masked_img, masked_img, raw_img = generate_img(path_img, type_img, size_img, crop_img, normalize_img, CV_NORMALIZE_TYPE[normalize_type])
        file_path = os.path.join(dest_path, class_name, f"{specie}-{disease}-{index}.jpg")
        specie_index = f"{specie}_{disease}_{index}"
        data = update_data_dict(data, 'classes', class_name)
        
        # FEATURES DEEP LEARNING
        if 'rgb' in answers_type_features or len(answers_type_features) == 0:
            data = update_data_dict(data, 'rgb_img', masked_img)
        if 'gabor' in answers_type_features:
            data = update_data_dict(data, 'gabor_img', get_gabor_img(masked_img))
        if 'gray' in answers_type_features:
            data = update_data_dict(data, 'gray_img', bgrtogray(masked_img))
        if 'canny' in answers_type_features:
            data = update_data_dict(data, 'canny_img', get_canny_img(masked_img))
        if 'lab' in answers_type_features:
            data = update_data_dict(data, 'lab',  get_lab_img(masked_img))
        if 'hsv' in answers_type_features:
            data = update_data_dict(data, 'hsv',  get_hsv_img(masked_img))
            
        # FEATURES MACHINE LEARNING
        if 'graycoprops' in answers_type_features:
            features = get_graycoprops(masked_img)
            for feature in features:
                data = update_data_dict(data, feature, features[feature])
        if 'lpb_histogram' in answers_type_features:
            features = get_lpb_histogram(masked_img)
            for feature in features:
                data = update_data_dict(data, feature, features[feature])
        if 'hue_moment' in answers_type_features:
            features = get_hue_moment(masked_img)
            for feature in features:
                data = update_data_dict(data, feature, features[feature])
        if 'haralick' in answers_type_features:
            features = get_haralick(masked_img)
            for feature in features:
                data = update_data_dict(data, feature, features[feature])
        if 'histogram_hsv' in answers_type_features:
            features = get_hsv_histogram(masked_img)
            for feature in features:
                data = update_data_dict(data, feature, features[feature])
        if 'histogram_lab' in answers_type_features:
            features = get_lab_histogram(masked_img)
            for feature in features:
                data = update_data_dict(data, feature, features[feature])
        if 'pyfeats' in answers_type_features and should_remove_bg:
            pyfeats_features = get_pyfeats_features(raw_img, mask)
            for feature in pyfeats_features:
                data = update_data_dict(data, feature, pyfeats_features[feature])
                
        if write_img:
            with safe_open_w(file_path) as f:
                pill_masked_img.save(f)
    return data


# MAIN
if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-a", "--augmented", required=False, action='store_true', default=False, help='Use directory augmented')
    parser.add_argument("-rmbg", "--remove-bg", required=False, action='store_true', default=False, help='Remove background before preprocess')
    parser.add_argument("-src", "--src-directory", required=False, type=str, default='', help='Directory source who can find images. default (data/{augmented})')
    parser.add_argument("-wi", "--write-img", required=False, action='store_true', default=False, help='Write images (png) in the new directory')
    parser.add_argument("-crop", "--crop-img", required=False, action='store_true', default=False, help='Remove padding around leaf')
    parser.add_argument("-nor", "--normalize-img", required=False, action='store_true', default=True, help='Normalize images, you can specify the normalization type with the option -nortype')
    parser.add_argument("-nortype", "--normalize-type", required=False, type=str, default='NORM_MINMAX', help='Normalize images features with cv.normalize (Default: NORM_MINMAX) \nTypes: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_NormTypes.html')
    parser.add_argument("-c", "--classification", required=False, type=str, default="HEALTHY_NOT_HEALTHY", help='Classification type: HEALTHY_NOT_HEALTHY(default), ONLY_HEALTHY, NOT_HEALTHY, ALL')
    parser.add_argument("-n", "--number-img", required=False, type=int, default=1000, help='Number of images to use per class to select maximum of all classes use -1. (default 1000)')
    parser.add_argument("-rt", "--result-type", required=False, type=str, default="GRAY", help='Type of result image for DP: GRAY, GABOR, CANNY, RGB. (default: GRAY)')
    parser.add_argument("-dst", "--destination", required=False, type=str, default='', help='Path to save the data. (default: data/preprocess)')
    parser.add_argument("-f", "--features", required=False, type=str, help='Features to extract separate by ","\nExample: -f=graycoprops,lpb_histogram,hue_moment\nList of features:\n   - For DP: rgb,gray,canny,gabor,lab,hsv\n   - For ML: graycoprops,lpb_histogram,hue_moment,haralick,histogram_hsv,histogram_lab,pyfeats')
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
    should_remove_bg = args.remove_bg
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
    iterator_species = tqdm(indexes_species, ncols=45) if VERBOSE else indexes_species
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(multiprocess_worker, iterator_species, repeat(df_filtred), repeat(data_used), repeat(type_output), repeat(src_directory), repeat(dest_path), repeat(size_img), repeat(crop_img), repeat(normalize_img), repeat(normalize_type), repeat(type_img), repeat(should_remove_bg), repeat(answers_type_features), repeat(write_img))
    data = dict()
    for result in results:
        for key, value in result.items():
            if key not in data.keys():
                print(key)
                print(value)
                data[key] = value if type(value) == list or type(value) == np.ndarray or type(value) == pd.DataFrame else [value]
            else:
                data[key] += value
    if VERBOSE:            
        for key in data.keys():
            print(f"[+] {key}: have len: {len(data[key])}")
                
    local_print(f"Total of images processed: {len(data['classes'])}")
    local_print(f"[+] Generate hdf5 file")
    prefix_data = 'all' if int(data_used) == -1 else str(data_used)
    path_hdf = os.path.join(dest_path, 'export', f"data_{type_output.lower()}_{prefix_data}_{'_'.join(answers_type_features)}.h5")
    os.makedirs(os.path.dirname(path_hdf), exist_ok=True)
    store_dataset(path_hdf, data, VERBOSE)
    local_print(f"[+] h5 file save at {path_hdf}")
