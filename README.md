# Vic 2 i

## Directories

```
analysis/
data/ -> Data sources
  augmentation/ -> Data augmented
  no_augmentation/ -> Data not augmented
preprocess/
utilities/
```

## Preprocess dataset

```
usage: main.py [-h] [-a] [-rmbg] [-src SRC_DIRECTORY] [-wi] [-crop] [-nor] [-nortype NORMALIZE_TYPE] [-c CLASSIFICATION] [-n NUMBER_IMG] [-rt RESULT_TYPE] [-dst DESTINATION] [-f FEATURES] [-s SIZE] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -a, --augmented       Use directory augmented
  -rmbg, --remove-bg    Remove background before preprocess
  -src SRC_DIRECTORY, --src-directory SRC_DIRECTORY
                        Directory source who can find images. default (data/{augmented})
  -wi, --write-img      Write images (png) in the new directory
  -crop, --crop-img     Remove padding around leaf
  -nor, --normalize-img
                        Normalize images, you can specify the normalization type with the option -nortype
  -nortype NORMALIZE_TYPE, --normalize-type NORMALIZE_TYPE
                        Normalize images features with cv.normalize (Default: NORM_MINMAX) 
                        Types: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_NormTypes.html
  -c CLASSIFICATION, --classification CLASSIFICATION
                        Classification type: HEALTHY_NOT_HEALTHY, ONLY_HEALTHY, NOT_HEALTHY, ALL (default)
  -n NUMBER_IMG, --number-img NUMBER_IMG
                        Number of images to use per class to select maximum of all classes use -1. (default 1000)
  -rt RESULT_TYPE, --result-type RESULT_TYPE
                        Type of result image for DP: GRAY, GABOR, CANNY, RGB. (default: GRAY)
  -dst DESTINATION, --destination DESTINATION
                        Path to save the data. (default: data/preprocess)
  -f FEATURES, --features FEATURES
                        Features to extract separate by ","
                        Example: -f=graycoprops,lpb_histogram,hue_moment
                        List of features:
                           - For DP: rgb,gray,canny,gabor,lab,hsv
                           - For ML: graycoprops,lpb_histogram,hue_moment,haralick,histogram_hsv,histogram_lab,pyfeats
  -s SIZE, --size SIZE  Size of images. (default 256x256)
  -v, --verbose         Verbose
```

# Web part

## With docker

```
cd app
docker compose up
```

## Manualy

### Start api
```
cd app/api
python run.py
```

### Start client
```
cd app/client/
yarn # install dependencies
yarn dev
```
