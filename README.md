# Leaf valley

## Directories

```
analysis/ -> data analysis in jupiter notebook
app/ -> web part
  api/
  client/
data/ -> Data sources
  augmentation/ -> Data augmented
  no_augmentation/ -> Data not augmented
preprocess/ -> scripts for preprocessing
process/ -> scripts for training
utilities/ -> global utilities functions
```

## Setup and installation

This script installs the dependencies for the project and the [dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1).

```
./setup.sh
```

## Preprocess dataset


This program allows you to preprocess the images, extract data and make a dataset in h5 format.

```
usage: python preprocess/main.py [-h] [-a] [-rmbg] [-src SRC_DIRECTORY] [-wi] [-crop] [-nor] [-nortype NORMALIZE_TYPE] [-c CLASSIFICATION] [-n NUMBER_IMG] [-rt RESULT_TYPE] [-dst DESTINATION] [-f FEATURES] [-s SIZE] [-v]

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

## Image augmentation


This program allows you to augment the images, and output this augmented images in the new directory.


```
usage: python preprocess/image_augmentation.py [-h] [-src SRC_DIRECTORY] [-dst DST_DIRECTORY] [-ncls NUMBER_IMG_BY_CLASS] [-naug NUMBER_IMG_AUGMENTATION] [-dup] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -src SRC_DIRECTORY, --src-directory SRC_DIRECTORY
                        Source of the directory where the images to augment are located. default (data/no_augmentation)
  -dst DST_DIRECTORY, --dst-directory DST_DIRECTORY
                        Source of the directory where the images to augment will be stored. default (data/tmp)
  -ncls NUMBER_IMG_BY_CLASS, --number-img-by-class NUMBER_IMG_BY_CLASS
                        Number of images to use per class to select maximum of all classes use -1. (default -1)
  -naug NUMBER_IMG_AUGMENTATION, --number-img-augmentation NUMBER_IMG_AUGMENTATION
                        Total number of augmented images, including original images. (default 3000)
  -dup, --remove-duplicate
                        Remove duplicate images created by the augmentaion process
  -v, --verbose         Verbose
```

## Deep Learning process


This program allows to train a deep learning model with the data preprocessed


```
python process/deep_learning/main.py -h
usage: process/deep_learning/main.py [-h] [-p PATH_DATASET] [-lt] [-es] [-b BATCH_SIZE] [-lr LEARNING_RATE] [-opt OPTIMIZER] [-e EPOCHS] [-m MODELS] [-s] [-dst-l DEST_LOGS] [-dst-m DEST_MODELS] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH_DATASET, --path-dataset PATH_DATASET
                        Path of your dataset (h5 file)
  -lt, --launch-tensorboard
                        Launch tensorboard after fitting
  -es, --early-stop     Early stop after fitting
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate (default 0.001)
  -opt OPTIMIZER, --optimizer OPTIMIZER
                        Optimizer (default adam). Available: dict_keys(['ADAM', 'RMSPROP', 'SGD', 'ADADELTA', 'NADAM'])
  -e EPOCHS, --epochs EPOCHS
                        Epoch
  -m MODELS, --models MODELS
                        Select model(s), if grid search is enabled, you can select multiple models separate by ",". example -m=vgg19,resnet50. By default is select all models.
                        Models availables:
                        VGG16,VGG16_PRETRAINED,VGG19,VGG19_PRETRAINED,RESNET50,RESNET50_PRETRAINED,CONVNEXTTINY,CONVNEXTTINY_PRETRAINED,CONVNEXTSMALL,CONVNEXTSMALL_PRETRAINED,CONVNEXTBASE,CONVNEXTBASE_PRETRAINED,CONVNEXTLARGE,CONVNEXTLARGE_PRETRAINED,RESNET50V2,RESNET50V2_PRETRAINED,INCEPTIONRESNETV2,INCEPTIONRESNETV2_PRETRAINED,INCEPTIONV3,INCEPTIONV3_PRETRAINED,EFFICIENTNETB0,EFFICIENTNETB0_PRETRAINED,EFFICIENTNETB7,EFFICIENTNETB7_PRETRAINED,XCEPTION,XCEPTION_PRETRAINED,CLASSIC_CNN,ALEXNET,LAB_PROCESS,LAB_INCEPTIONV3_PROCESS,HSV_PROCESS,GOOGLE/VIT-BASE-PATCH16,GOOGLE/VIT-BASE-PATCH32,GOOGLE/VIT-LARGE-PATCH16,GOOGLE/VIT-LARGE-PATCH32,FACEBOOK/CONVNEXT-BASE,FACEBOOK/CONVNEXT-LARGE,FACEBOOK/CONVNEXT-XLARGE.
  -s, --save-model      Save model
  -dst-l DEST_LOGS, --dest-logs DEST_LOGS
                        Destination for tensorboard logs. (default logs/tensorboard)
  -dst-m DEST_MODELS, --dest-models DEST_MODELS
                        Destination for model if save model is enabled
  -v, --verbose         Verbose
```

<details>
  <summary><strong>Deep learning models availables:</strong></summary>
<br>
<h6>VGG</h6>

  - <small>VGG16</small>
  - <small>VGG19</small>
<h6>ResNet</h6>

  - <small>RESNET50</small>

<h6>Convnext</h6>

  - <small>CONVNEXTTINY</small>
  - <small>CONVNEXTSMALL</small>
  - <small>CONVNEXTBASE</small>
  - <small>CONVNEXTLARGE</small>

<h6>ResNet & Inception & Xception</h6>

  - <small>RESNET50V2</small>
  - <small>INCEPTIONRESNETV2</small>
  - <small>INCEPTIONV3</small>
  - <small>XCEPTION</small>

<h6>EfficientNet</h6>

  - <small>EFFICIENTNETB0</small>
  - <small>EFFICIENTNETB7</small>

<h6>Transformers</h6>

  - <small>GOOGLE/VIT-BASE-PATCH16</small>
  - <small>GOOGLE/VIT-BASE-PATCH32</small>
  - <small>GOOGLE/VIT-LARGE-PATCH16</small>
  - <small>GOOGLE/VIT-LARGE-PATCH32</small>
  - <small>FACEBOOK/CONVNEXT-BASE</small>
  - <small>FACEBOOK/CONVNEXT-LARGE</small>
  - <small>FACEBOOK/CONVNEXT-XLARGE</small>


<h6>Homemade models</h6>

  - <small>CLASSIC_CNN</small>
  - <small>ALEXNET</small>
  - <small>LAB_PROCESS</small>
  - <small>LAB_INCEPTIONV3_PROCESS</small>
  - <small>HSV_PROCESS</small>
</details>

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
pip install -r requirements.txt
python run.py
```

### Start client
```
cd app/client/
yarn
yarn dev
```
