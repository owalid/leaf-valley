'''
  CLI used to preprocess the data and get features and classes.
'''

import random
import os
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
from utilities.utils import crop_resize_image, safe_open_w, get_df, get_canny_img, get_gabor_img, store_dataset, update_data_dict
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


def generate_img_without_bg(src_directory, img_number, type_img, size_img, cropped_img=False):
    path_img = f"{src_directory}/image ({img_number}).JPG"
    bgr_img, _, _ = pcv.readimage(path_img, mode='bgr')

    mask, new_img = remove_bg(bgr_img)

    if cropped_img == True:
        new_img = crop_resize_image(new_img, new_img)

    im = Image.fromarray(new_img)
    enhancer = ImageEnhance.Sharpness(im)
    im_s_1 = enhancer.enhance(2)

    if type_img == 'canny':
        edges = pcv.canny_edge_detect(np.array(im_s_1))
        normalized_masked_img = cv.normalize(
            edges, edges, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        return Image.fromarray(edges), cv.resize(normalized_masked_img, size_img), edges
    else:  # COLOR OR GRAY
        array_img = np.array(im_s_1)
        array_img = cv.resize(array_img, size_img)
        normalized_masked_img = cv.normalize(
            array_img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        if type_img == 'gray':
            gray_img = cv.cvtColor(array_img, cv.COLOR_BGR2GRAY)
            im_s_1 = Image.fromarray(gray_img)

        return im_s_1, normalized_masked_img, array_img, bgr_img, mask


# MAIN
if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-a", "--augmented", required=False, action='store_true', default=False, help='Use directory augmented')
    parser.add_argument("-src", "--src-directory", required=False, type=str, default='', help='Directory source who can find images. default (data/{augmented})')
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
    res_augmented = 'augmentation' if args.augmented else 'no_augmentation'
    src_directory = os.path.join(args.src_directory, res_augmented) if args.src_directory != '' else f"data/{res_augmented}"
    df = get_df(src_directory)
    type_output = args.classification
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
    answers_type_features = [type_img, 'graycoprops', 'lpb_histogram', 'hue_moment', 'haralick', 'histogram_hsv', 'histogram_lab', 'pyfeats'] if len(answers_type_features) == 0 else answers_type_features
    
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
    if len(answers_type_features) > 0:
        local_print(f"[+] answers_type_features: {answers_type_features}")
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
        
        if current_data_used == -1 or current_df.number_img < current_data_used:
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
            file_path = f"{dest_path}/{label}/{specie}-{disease}-{index}.jpg"
            pill_masked_img, normalized_masked_img, masked_img, raw_img, mask = generate_img_without_bg(
                f"{src_directory}/{specie_directory}", index, type_img, size_img)
            specie_index = f"{specie}_{disease}_{index}"
            data = dict()
            data['label'] = specie_index
            data['class'] = label
            
                
            # FEATURES DEEP LEARNING
            if 'rgb' in answers_type_features or len(answers_type_features) == 0:
                data['rgb_img'].append(normalized_masked_img)
            if 'gabor' in answers_type_features:
                data['gabor_img'].append(get_gabor_img(normalized_masked_img))
            if 'gray' in answers_type_features:
                gray_img = cv.cvtColor(normalized_masked_img, cv.COLOR_BGR2GRAY)
                data['gray_img'].append(gray_img)
            if 'canny' in answers_type_features:
                data['canny_img'].append(get_canny_img(normalized_masked_img))
                
            # FEATURES MACHINE LEARNING
            if 'graycoprops' in answers_type_features:
                data['graycoprops'] = get_graycoprops(masked_img)
            if 'lpb_histogram' in answers_type_features:
                data['lpb_histogram'] = get_lbp_histogram(normalized_masked_img)
            if 'hue_moment' in answers_type_features:
                data['hue_moment'] = get_hue_moment(normalized_masked_img)
            if 'haralick' in answers_type_features:
                data['haralick'] = get_haralick(normalized_masked_img)
            if 'histogram_hsv' in answers_type_features:
                data['histogram_hsv'] = get_hsv_histogram(normalized_masked_img)
            if 'histogram_lab' in answers_type_features:
                data['histogram_lab'] = get_lab_histogram(normalized_masked_img)
            if 'pyfeats' in answers_type_features:
                data.update(get_pyfeats_features(raw_img, mask))
                    
            series.append(pd.Series(data))
            with safe_open_w(file_path) as f:
                pill_masked_img.save(f)
        local_print(f"[+] End with {label}\n\n")
    df_features = pd.DataFrame(series)

    local_print(f"Number of images, {len(df_features)}")
    local_print(f"[+] Generate pickle")
    prefix_data = 'all' if int(data_used) == -1 else str(data_used)
    path_pickle = f"{dest_path}/export/data_{type_output.lower()}_{prefix_data}_{type_img}.pkl"
    os.makedirs(os.path.dirname(path_pickle), exist_ok=True)
    joblib.dump(df_features, path_pickle)
    local_print(f"[+] pickle save at {path_pickle}")
