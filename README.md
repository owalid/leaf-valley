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
python preprocess/main.py -h
usage: main.py [-h] [-a] [-src SRC_DIRECTORY] [-c CLASSIFICATION] [-n NUMBER_IMG] [-rt RESULT_TYPE] [-dst DESTINATION] [-f FEATURES] [-s SIZE] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -a, --augmented       Use directory augmented
  -src SRC_DIRECTORY, --src-directory SRC_DIRECTORY
                        Directory source who can find images. default (data/{augmented})
  -c CLASSIFICATION, --classification CLASSIFICATION
                        Classification type: HEALTHY_NOT_HEALTHY(default), ONLY_HEALTHY, NOT_HEALTHY, ALL
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
                           - For DP: rgb, gray, canny, gabor
                           - For ML: graycoprops, lpb_histogram, hue_moment, haralick, histogram_hsv, histogram_lab, pyfeats
  -s SIZE, --size SIZE  Size of images. (default 256x256)
  -v, --verbose         Verbose
```
