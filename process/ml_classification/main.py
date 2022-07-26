'''
  CLI used to manage ML classification.
'''
import argparse as ap
import random

import load_data_from_h5


# MAIN
if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument("-src", "--src-directory", required=False, type=str, default='',
                        help='Directory source who can find images. default: data/preprocess/ml_classification')
    parser.add_argument("-f", "--filename", required=True, type=str, default='',
                        help='Filename input data format h5 for raw data or pkl for train/test data')
    parser.add_argument("-dst", "--destination", required=False, type=str,
                        default='', help='Path to save the data. default: data/preprocess/ml_classification')
    parser.add_argument("-sd", "--save-data", required=False,
                        action='store_true', default=False, help='Save train/test data and options_datasets')
    parser.add_argument("-nortype", "--normalize-type", required=False, type=str, default='NORM_MINMAX',
                        help='Normalize data (NORM_STANDERDEV or NORM_MINMAX normalization) (Default: NORM_MINMAX)')
    parser.add_argument("-cs", "--classification-step", required=False, type=str, default="ALL",
                        help='Classification step: LOAD_DATA, FIT_MODEL, PREDICT_MODEL, FIT_PREDICT_MODEL, ALL (default)')
    parser.add_argument("-m", "--classification-models", required=False, type=str, default="ALL",
                        help='Classification step: LOAD_DATA, FIT_MODEL, PREDICT_MODEL, FIT_PREDICT_MODEL, ALL (default)')
    parser.add_argument("-sm", "--save-model", required=False,
                        action='store_true', default=True, help='Save model')
    parser.add_argument("-sp", "--save-plot", required=False,
                        action='store_true', default=True, help='Save heatmap plot')
    parser.add_argument("-ss", "--save-scores", required=False,
                        action='store_true', default=True, help='Save scores')

    parser.add_argument("-rmbg", "--remove-bg", required=False, action='store_true',
                        default=False, help='Remove background before preprocess')
    parser.add_argument("-wi", "--write-img", required=False, action='store_true',
                        default=False, help='Write images (png) in the new directory')
    parser.add_argument("-crop", "--crop-img", required=False,
                        action='store_true', default=False, help='Remove padding around leaf')
    parser.add_argument("-n", "--number-img", required=False, type=int, default=1000,
                        help='Number of images to use per class to select maximum of all classes use -1. (default 1000)')
    parser.add_argument("-rt", "--result-type", required=False, type=str, default="GRAY",
                        help='Type of result image for DP: GRAY, GABOR, CANNY, RGB. (default: GRAY)')
    parser.add_argument("-f", "--features", required=False, type=str,
                        help='Features to extract separate by ","\nExample: -f=graycoprops,lpb_histogram,hue_moment\nList of features:\n   - For DP: rgb,gray,canny,gabor,lab,hsv\n   - For ML: graycoprops,lpb_histogram,hue_moment,haralick,histogram_hsv,histogram_lab,pyfeats')
    parser.add_argument("-s", "--size", required=False, type=int,
                        default=256, help='Size of images. (default 256x256)')
    parser.add_argument("-v", "--verbose", required=False,
                        action='store_true', default=False, help='Verbose')
    args = parser.parse_args()
    print(args)

    random.seed(42)

