from unet import unet_model
import tensorflow as tf
import argparse as ap
import os
from argparse import RawTextHelpFormatter
from pathlib import Path
import imghdr

VERBOSE = False
DATASET_PATH = 'data/unet_dataset'


def local_print(msg):
    if VERBOSE:
        print(msg)

def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (256, 256), method='nearest')
    input_mask = tf.image.resize(mask, (256, 256), method='nearest')

    return input_image, input_mask

def check_images(data_dir, ext_list):
    result = []
    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
      if filepath.suffix.lower() in ext_list:
        img_type = imghdr.what(filepath)
        if img_type is None or img_type not in img_type_accepted_by_tf:
            result.append(filepath)
    return result

def check_dataset():
     if os.path.exists(DATASET_PATH):
        local_print('[+] Dataset directory found')
        dataset_lst = os.listdir(DATASET_PATH)
        if len(dataset_lst) == 2 and 'rgb' in dataset_lst and 'mask' in dataset_lst:
            local_print('[+] /mask and /rgb directories found')
            
            rgb_lst = os.listdir(path_rgb)
            mask_lst = os.listdir(path_mask)

            if len(rgb_lst) == len(mask_lst):
                local_print('[+] Found {} images in /rgb and /mask directories'.format(len(rgb_lst)))
                bad_images_rgb = check_images(path_rgb, 'png')
                bad_images_mask = check_images(path_mask, 'png')

                if len(bad_images_rgb) == 0:
                    local_print('[+] No bad images found in /rgb directory')
                else:
                    print(f'[-] Found images with unsupported format in /rgb directory\n{bad_images_rgb}')
                    exit(1)

                if len(bad_images_mask) == 0:
                    local_print('[+] No bad images found in /mask directory')
                else:
                    print(f'[-] Found images with unsupported format in /mask directory\n{bad_images_mask}')
                    exit(2)

            else:
                print('[-] /rgb and /mask directories have different number of images')
                exit(3)
        else:
            print('[-] Please don\'t rename the dataset directory')
            exit(4)

if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-dst", "--dest-model", required=False, type=str, default='models/segmentation/unet.h5', help='Destination directory model (default: models/segmentation/unet.h5)')
    parser.add_argument("-e", "--epoch", required=False, type=int, default=40, help='Epoch (default 40)')
    parser.add_argument("-bs", "--batch-size", required=False, type=int, default=32, help='Batch size (default 32)')
    parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help='Verbose')
    args = parser.parse_args()
    print(args)

    dest_path = args.dest_model if args.dest_model is None else 'models/segmentation'
    epoch = int(args.epoch) if int(args.epoch) else 40
    batch_size = int(args.batch_size) if int(args.batch_size) else 32
    verbose = args.verbose

    check_dataset() # Check if dataset exists and has the right structure

    path_rgb = os.path.join(DATASET_PATH, 'rgb')
    path_mask = os.path.join(DATASET_PATH, 'mask')


    if not os.path.exists(dest_path):
        local_print('[+] Creating model directory')
        os.makedirs(dest_path)

    image_list = os.listdir(path_rgb)
    mask_list = os.listdir(path_mask)
    image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
    mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)
    image_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)

    unet = unet_model((256, 256, 3))
    local_print(unet.summary())
    unet.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    processed_image_ds.batch(batch_size)
    train_dataset = processed_image_ds.cache().shuffle(500).batch(batch_size)
    model_history = unet.fit(train_dataset, epochs=epoch)
    final_path_unet = os.path.join(dest_path, 'unet.h5')
    unet.save(final_path_unet)
    local_print(f"[+] Model saved to {final_path_unet}")
