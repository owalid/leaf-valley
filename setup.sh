#/bin/bash

# Download dataset
mkdir data
echo "Downloading dataset..."
curl -o data/datasets.zip "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/tywbtsjrjv-1.zip"
unzip -q data/datasets.zip -d data/
rm data/datasets.zip
unzip -q data/\*.zip -d data/
mv data/Plant_leave_diseases_dataset_with_augmentation data/augmentation
mv data/Plant_leave_diseases_dataset_without_augmentation data/no_augmentation
rm data/*.zip

# Install dependencies
echo "Install python dependencies"
pip install -r requirements.txt

# Create dataset for unet segmentation
echo "Create dataset for unet segmentation"
python preprocess/unet_segmentation/generate_dataset.py
