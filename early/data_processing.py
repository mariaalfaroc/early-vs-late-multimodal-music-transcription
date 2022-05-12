# -*- coding: utf-8 -*-

import os, json, random

from typing import Tuple

import cv2
import numpy as np

import config

# Get all the filenames of the corresponding data folds partition
# Ex.: if fold_type = "test" -> folds = [".../test_gt_fold0.dat", ".../test_gt_fold1.dat", ...]
def get_folds_filenames(fold_type: str) -> list:
    folds = []
    for fname in os.listdir(config.folds_dir):
        if fname.startswith(fold_type):
            folds.append(os.path.join(config.folds_dir, fname))
    return sorted(folds)

# Get all images and labels filenames 
# They are nested lists where element number X in the list contains the filenames used in fold number X
def get_datafolds_filenames(folds_files: list) -> Tuple[list, list, list]:
    omr_images_filenames = []
    amt_images_filenames = []
    labels_filenames = []
    # Iterate over each data fold file
    for filename in folds_files:
        with open(filename) as f:
            lines = f.read().splitlines()
        omr_images_filenames.append([os.path.join(config.omr_images_dir, fname + config.omr_image_extn) for fname in lines])
        amt_images_filenames.append([os.path.join(config.amt_images_dir, fname + config.amt_image_extn) for fname in lines])
        labels_filenames.append([os.path.join(config.labels_dir, fname + config.label_extn) for fname in lines])
    return omr_images_filenames, amt_images_filenames, labels_filenames

# --------------------

# Get dictionaries for w2i and i2w conversion correspoding to a single training fold
def get_fold_vocabularies(train_labels_fnames: list) -> Tuple[dict, dict]:
    # Get all tokens related to a SINGLE train data fold
    tokens = []
    for fname in train_labels_fnames:
        with open(fname) as f:
            tokens.extend(f.read().split())
    # Eliminate duplicates and sort them alphabetically
    tokens = sorted(set(tokens))
    # Create vocabularies
    w2i = dict(zip(tokens, range(len(tokens))))
    i2w = dict(zip(range(len(tokens)), tokens))
    return w2i, i2w

# Utility function for saving w2i dictionary in a JSON file
def save_w2i_dictionary(w2i, filepath):
    # Save w2i dictionary to JSON filepath to later retrieve it
    # No need to save both of them as they are related
    with open(filepath, "w") as json_file:
        json.dump(w2i, json_file)
    return 

# Retrieve w2i and i2w dictionaries from w2i JSON file
def load_dictionaries(filepath) -> Tuple[dict, dict]:
    with open(filepath, "r") as json_file:
        w2i = json.load(json_file)
    i2w = {int(v): k for k, v in w2i.items()}
    return w2i, i2w

# --------------------

# Preprocess image (read from path, convert to grayscale, normalize, and resize preserving aspect ratio)
def preprocess_image(image_path, image_flag):
    img = cv2.imread(image_path, image_flag)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (255. - img) / 255.
    new_height = config.img_max_height
    new_width = int(new_height * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    img = img.reshape(new_height, new_width, 1)
    return img, img.shape[1] // config.width_reduction

# Preprocess label (read from path, split by encoding grammar, and convert to integer if training)
def preprocess_label(label_path, training, w2i):
    label = open(label_path).read().split()
    if training:
        label = [w2i[w] for w in label]
        return label, len(label)
    return label

# CTC-loss preprocess function -> (xi, yi)
# xi[0] -> omr images, zero-padded to the maximum image width found
# xi[1] -> amt images, zero-padded to the maximum image width found
# xi[2] -> real width (after the CNN) of the images
# xi[3] -> labels, CTC-blank-padded to the maximum label length found
# xi[4] -> real length of the labels
# yi[0] -> dummy value for CTC-loss 
def ctc_preprocess(omr_images: list, amt_images: list, labels: list, blank_index: int) -> Tuple[dict, dict]:
    # Unzip variables
    omr_images, omr_images_len = list(zip(*omr_images))
    amt_images, amt_images_len = list(zip(*amt_images))
    labels, labels_len = list(zip(*labels))
    # Obtain the current batch size
    num_samples = len(omr_images)
    # Zero-pad images to maximum batch image width
    max_width = max(omr_images + amt_images, key=np.shape).shape[1]
    omr_images = np.array([np.pad(i, pad_width=((0, 0), (0, max_width - i.shape[1]), (0, 0))) for i in omr_images], dtype="float32")
    amt_images = np.array([np.pad(i, pad_width=((0, 0), (0, max_width - i.shape[1]), (0, 0))) for i in amt_images], dtype="float32")
    images_len = [max(i_len, a_len) for i_len, a_len in zip(omr_images_len, amt_images_len)]
    images_len = np.array(images_len, dtype="int32").reshape((num_samples, 1))
    # CTC-blank-pad labels to maximum batch label length
    max_length = len(max(labels, key=len))
    labels = np.array([i + [blank_index] * (max_length - len(i)) for i in labels], dtype="int32")
    labels_len = np.array(labels_len, dtype="int32").reshape((num_samples, 1))
    # Format data
    xi = {
        "omr_image": omr_images,
        "amt_image": amt_images,
        "image_len": images_len, 
        "label": labels, 
        "label_len": labels_len
    }
    yi = {
        "ctc_loss": np.zeros(shape=(num_samples, 1), dtype="float32")
    }
    return xi, yi

# Train data generator
def train_data_generator(omr_images_files: list, amt_images_files: list, labels_files: list, w2i: dict) -> Tuple[dict, dict]:
    data = list(zip(omr_images_files, amt_images_files, labels_files))
    random.shuffle(data)
    omr_images_files, amt_images_files, labels_files = zip(*data)
    del data
    size = len(omr_images_files)
    start = 0
    while True:
        end = min(start + config.batch_size, size)
        omr_images, amt_images, labels = [preprocess_image(i, config.omr_image_flag) for i in omr_images_files[start:end]], [preprocess_image(i, config.amt_image_flag) for i in amt_images_files[start:end]], [preprocess_label(i, training=True, w2i=w2i) for i in labels_files[start:end]]
        xi, yi = ctc_preprocess(omr_images, amt_images, labels, blank_index=len(w2i))
        if end == size:
            start = 0
            # Due to the current training set up (model.fit() is called one epoch at a time), 
            # it does not make sense to shuffle the data in this step
            # as the generator is stopped after all the training data is seen
            # This is why it is important to shuffle the data at the very beginning,
            # so that at each epoch it is seen in a different order
            # Uncomment the following lines, if we were to call model.fit() for longer that one epoch
            # data = list(zip(omr_images_files, amt_images_files, labels_files))
            # random.shuffle(data)
            # omr_images_files, amt_images_files, labels_files = zip(*data)
            # del data
        else:
            start = end
        yield xi, yi
