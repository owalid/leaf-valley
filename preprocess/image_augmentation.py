### `Image augmentation`

import os
import shutil
import random as rd
import argparse as ap
from tqdm import tqdm
import concurrent.futures
from itertools import repeat
import Augmentor
from difPy import dif
from datetime import datetime as dt 


def local_print(msg):
    if VERBOSE:
        print(msg)

def images_augmentation_pipline(dir, num_of_samples):
    try:
        # Create a pipeline
        p = Augmentor.Pipeline(dir, output_directory='.')

        # Add operations to a pipeline.
        p.flip_random(probability=.3) # Add a random flip operation to the pipeline:
        p.rotate_random_90(probability=.1) 
        p.skew(probability=.6) # Perspective Skewing
        p.random_distortion(probability=.3, grid_height=4, grid_width=4, magnitude=8) # Elastic Distortions
        p.shear(probability=.3, max_shear_left=20, max_shear_right=20) # Shearing
        p.crop_random(probability=.1, percentage_area=.85, randomise_percentage_area=False) # Cropping

        p.sample(num_of_samples - len(os.listdir(dir)))

        # Remove duplicate images
        if remove_duplicate:
            search = dif(dir, similarity='high', delete=True, silent_del=True)
            local_print(search.result)

        return False
    except Exception as e:
        return e

def images_augmentation_process(path_in, path_out, num_of_samples, NByCls=-1):
    for d in plants:
        if os.path.exists(os.path.join(path_out,d)):
            shutil.rmtree(os.path.join(path_out,d))
        os.makedirs(os.path.join(path_out,d))

        img_lst = []
        for f in os.listdir(os.path.join(path_in,d)):
            img_lst.append(os.path.join(d,f))
        img_lst = rd.sample(img_lst, len(img_lst) if (NByCls==-1)|(NByCls>len(img_lst)) else NByCls)
        rd.shuffle(img_lst)

        local_print(f'Copy {"all" if NByCls==-1 else NByCls} images from {os.path.join(path_in,d)}\t\tto ==> {os.path.join(path_out,d)}')      
        for f in img_lst:
            shutil.copy(os.path.join(path_in,f),os.path.join(path_out,d))


    # Images augmentation
    start = dt.now()
    local_print('Images augmentation stared at : {start}')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        if VERBOSE:
            result = list(tqdm(executor.map(images_augmentation_pipline, [os.path.join(path_out,d) for d in plants], repeat(num_of_samples)), total=len(plants)))
        else:
            result = executor.map(images_augmentation_pipline, [os.path.join(path_out,d) for d in plants], repeat(num_of_samples))

    for r in result:
        if r!=False:
            print(r)
        
    local_print(f'\n============\nImages Augmtentation took {dt.now()-start} ================')

if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument("-src", "--src-directory", required=False, type=str, default='data/no_augmentation',
                        help='Source of the directory where the images to augment are located. default (data/no_augmentation)')
    parser.add_argument("-dst", "--dst-directory", required=False, type=str, default='data/tmp',
                        help='Source of the directory where the images to augment will be stored. default (data/tmp)')
    parser.add_argument("-ncls", "--number-img-by-class", required=False, type=int, default=-1,
                        help='Number of images to use per class to select maximum of all classes use -1. (default -1)')
    parser.add_argument("-naug", "--number-img-augmentation", required=False, type=int, default=3000,
                        help='Total number of augmented images, including original images. (default 3000)')
    parser.add_argument("-dup", "--remove-duplicate", required=False,
                        action='store_true', default=False, help='Remove duplicate images created by the augmentaion process')
    parser.add_argument("-v", "--verbose", required=False,
                        action='store_true', default=False, help='Verbose')
    args = parser.parse_args()

    VERBOSE = args.verbose
    remove_duplicate = args.remove_duplicate

    print("\n=====================================================")
    print(f"[+] Data input directory  : {args.src_directory}")
    print(f"[+] Data output directory : {args.dst_directory}")
    print(f"[+] Total number of output images : {args.number_img_augmentation}")
    print(f"[+] Number of images by class for the input : {args.number_img_by_class}")
    print(f"[+] Remove duplicate images created by the augmentation process : {args.remove_duplicate}")
    print(f"[+] VERBOSE: {VERBOSE}")

    plants = [d for d in os.listdir(args.src_directory) if (os.path.isdir(os.path.join(args.src_directory,d))) and (d!='Background_without_leaves')]

    images_augmentation_process(args.src_directory, args.dst_directory, args.number_img_augmentation, args.number_img_by_class)