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

## Deep Learning process

```
python process/deep_learning/main.py -h
usage: main.py [-h] [-p PATH_DATASET] [-lt] [-b BATCH_SIZE] [-e EPOCHS] [-m MODELS] [-s] [-dst-l DEST_LOGS] [-dst-m DEST_MODELS] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH_DATASET, --path-dataset PATH_DATASET
                        Path of your dataset (h5 file)
  -lt, --launch-tensorboard
                        Launch tensorboard after fitting
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -e EPOCHS, --epochs EPOCHS
                        Epoch
  -m MODELS, --models MODELS
                        Select model(s), if grid search is enabled, you can select multiple models separate by ",". example -m=vgg19,resnet50. By default is select all models.
                        Models availables:
                        VGG16,VGG16_PRETRAINED,VGG19,VGG19_PRETRAINED,RESNET50,RESNET50_PRETRAINED,RESNET50V2,RESNET50V2_PRETRAINED,INCEPTIONRESNETV2,INCEPTIONRESNETV2_PRETRAINED,INCEPTIONV3,INCEPTIONV3_PRETRAINED,EFFICIENTNETB0,EFFICIENTNETB0_PRETRAINED,EFFICIENTNETB7,EFFICIENTNETB7_PRETRAINED,XCEPTION,XCEPTION_PRETRAINED,CLASSIC_CNN,ALEXNET,LAB_PROCESS,HSV_PROCESS.
  -s, --save-model      Save model
  -dst-l DEST_LOGS, --dest-logs DEST_LOGS
                        Destination for tensorboard logs. (default logs/tensorboard)
  -dst-m DEST_MODELS, --dest-models DEST_MODELS
                        Destination for model if save model is enabled
  -v, --verbose         Verbose
```

# Web part

## With docker

```
cd app
docker compose up
```

## Manualy


# Web part

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
