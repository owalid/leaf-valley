'''
  CLI used to preprocess the data and get features and classes.
'''

import os
import cv2 as cv
import joblib
import numpy as np
from plantcv import plantcv as pcv
from PIL import Image, ImageEnhance
import pandas as pd
import sys
sys.path.append("../utilities")
from utils import crop_resize_image, safe_open_w, get_df, get_canny_img, get_gabor_img
from extract_features import get_pyfeats_features, get_lbp_histogram, get_hue_moment, get_haralick, get_hsv_histogram, get_lab_histogram, get_graycoprops
from remove_background_functions import remove_bg

pcv.params.debug = ''
debug = ''

DEFAULT_FINAL_IMG_SIZE = (256, 256)

HEALTHY_NOT_HEALTHY = 'HEALTHY_NOT_HEALTHY'
ONLY_HEALTHY = 'ONLY_HEALTHY'
NOT_HEALTHY = 'NOT_HEALTHY'
ALL = 'ALL'


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


def generate_img_without_bg(specie_directory, img_number, type_img, specie, healthy, size_img, cropped_img=False):
    path_img = f"../data/augmentation/{specie_directory}/image ({img_number}).JPG"
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
    res_augmented = input(
        'use not augmented or augmented ?\n0) for not augmented (default)\n1) for augmented\n> ')

    if res_augmented == '1':
        res_augmented = 'augmented'
        df = get_df('../data/augmentation')
    else:
        res_augmented = 'no_augmented'
        df = get_df('../data/no_augmentation')

    type_output = input(
        'Type of outputs \n0) healthy_not_healthy (default)\n1) only healthy\n2) only disases\n> ')
    if type_output == '' or not type_output.isnumeric() or int(type_output) == 0:
        type_output = HEALTHY_NOT_HEALTHY
    elif int(type_output) == 1:
        type_output = ONLY_HEALTHY
    else:
        type_output = NOT_HEALTHY

    df_filtred = get_df_filtered(df, type_output)
    indexes_species = df_filtred.index

    data_used_raw = int(input(
        'Choose number of images per class maximum 1000 (default maximum per class)\n> '))
    data_used = get_data_used(data_used_raw, df_filtred, type_output)

    type_img_code = input(
        f'Choose your result type image: \n0) for canny_edge\n1) for gray_scale (default)\n2) for rgb\n> ')
    if type_img_code == '0':
        type_img = 'canny'
    elif type_img_code == '2':
        type_img = 'color'
    else:
        type_img = 'gray'

    default_path_result = f"../data/preprocess/{type_output}/{res_augmented}/{type_img}"
    choose_final_path = input(
        f'Choose your final path \n(default is: {default_path_result}/<specie>/<number>.jpg)\n> ')
    choose_final_path = default_path_result if choose_final_path == '' else choose_final_path

    generate_pickle = input(f'Want to generate pickle (default Y) ? Y/n\n> ')
    generate_pickle = 'y' if generate_pickle == '' else generate_pickle.lower()
    answers_type_features = []

    if generate_pickle == 'y':
        type_features = [
            inquirer.Checkbox(
                "type",
                message="Choose your features in pickles:",
                choices=["rgb", "gray", "canny", "gabor", "graycoprops", "lpb_histogram",
                         'hue_moment', 'haralick', 'histogram_hsv', 'histogram_lab', 'pyfeats'],
            ),
        ]
        answers_type_features = inquirer.prompt(type_features)
        answers_type_features = answers_type_features['type']

    size_img = input(f'Size of output images (default 256px) ?\n> ')
    size_img = DEFAULT_FINAL_IMG_SIZE if size_img == '' or not size_img.isnumeric(
    ) else (int(size_img), int(size_img))

    data = dict()
    data = {x: [] for x in answers_type_features}
    data['label'] = []
    # print(df_features)
    print('\n')
    print("=====================================================")
    print(f"[+] type dataset: {res_augmented}")
    print(f"[+] type output: {type_output}")
    print(f"[+] size output images: {size_img}")
    print(f"[+] type image: {type_img}")
    print(f"[+] path: {choose_final_path}")
    if len(answers_type_features) > 0:
        print(f"[+] answers_type_features: {answers_type_features}")
    print("=====================================================")
    for specie_directory in indexes_species:
        current_df = df_filtred.loc[specie_directory]
        healthy = current_df.healthy
        disease = current_df.disease
        specie = current_df.specie

        current_data_used = data_used if not isinstance(
            data_used, dict) else data_used['healthy' if healthy else 'not_healthy']
        number_img = current_df.number_img if current_data_used == - \
            1 or current_df.number_img < current_data_used else current_data_used
        print(f"[+] index {specie_directory}")
        print(f"[+] Start generate specie: {specie}")
        print(f"[+] Number of images: {number_img}")

        if type_output == HEALTHY_NOT_HEALTHY:
            label = 'healthy' if healthy else 'not_healthy'
        elif type_output == NOT_HEALTHY:
            label = disease
        else:
            label = specie
        for index in range(1, number_img):
            if int(number_img / 2) == index:
                print("[+] 50%")
            pill_masked_img, normalized_masked_img, masked_img, raw_img, mask = generate_img_without_bg(
                specie_directory, index, type_img, specie, healthy, size_img)
            file_path = f"{choose_final_path}/{label}/{specie}-{disease}-{index}.jpg"
            specie_index = f"{specie}_{disease}_{index}"
            # df_features.loc[specie_index] = {}
            # df_features[specie_index] = {}
            data['label'].append(specie_index)
            
            data
            if generate_pickle.lower() == 'y':
                if 'rgb' in answers_type_features or len(answers_type_features) == 0:
                    data['rgb_img'].append(normalized_masked_img)
                if 'gabor' in answers_type_features:
                    data['gabor_img'].append(get_gabor_img(normalized_masked_img))
                if 'gray' in answers_type_features:
                    gray_img = cv.cvtColor(normalized_masked_img, cv.COLOR_BGR2GRAY)
                    data['gray_img'].append(gray_img)
                if 'canny' in answers_type_features:
                    data['canny_img'].append(get_canny_img(normalized_masked_img))

                if 'graycoprops' in answers_type_features:
                    print("get_graycoprops", list(get_graycoprops(masked_img)))
                    data['graycoprops'].append(list(get_graycoprops(masked_img)))
                if 'lpb_histogram' in answers_type_features:
                    data['lpb_histogram'].append(get_lbp_histogram(normalized_masked_img))
                if 'hue_moment' in answers_type_features:
                    data['hue_moment'].append(get_hue_moment(normalized_masked_img))
                if 'haralick' in answers_type_features:
                    data['haralick'].append(get_haralick(normalized_masked_img))
                if 'histogram_hsv' in answers_type_features:
                    data['histogram_hsv'].append(get_hsv_histogram(normalized_masked_img))
                if 'histogram_lab' in answers_type_features:
                    data['histogram_lab'].append(get_lab_histogram(normalized_masked_img))
                if 'pyfeats' in answers_type_features:
                    tmpDf = pd.DataFrame({})
                    get_pyfeats_features(tmpDf, specie_index, raw_img, mask)
                    # data['pyfeats'].append(get_pyfeats_features(raw_img, mask))
            # df_features.concat([df_features, temporary_df])
            # df_features.update(temporary_df)
            with safe_open_w(file_path) as f:
                pill_masked_img.save(f)
        print(f"[+] End with {label}\n\n")
    # print(df_features)

    if generate_pickle.lower() == 'y':
        print(f"Number of images, {len(data)}")
        print(f"[+] Generate pickle")
        prefix_data = 'all' if int(data_used_raw) == -1 else str(data_used_raw)
        path_pickle = f"{choose_final_path}/export/data_{type_output.lower()}_{prefix_data}_{type_img}.pkl"
        os.makedirs(os.path.dirname(path_pickle), exist_ok=True)
        joblib.dump(data, path_pickle)
        print(f"[+] pickle save at {path_pickle}")
