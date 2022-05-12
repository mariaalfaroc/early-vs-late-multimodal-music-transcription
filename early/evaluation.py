# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from data_processing import preprocess_image, preprocess_label

# CTC-greedy decoder (merge repeated elements, remove blank labels, and covert back to string labels)
def ctc_greedy_decoder(y_pred: tf.Tensor, input_length: list, i2w: dict) -> list:
    input_length = tf.constant(input_length, dtype="int32", shape=(len(input_length),))
    # Blank labels are returned as -1
    y_pred = keras.backend.ctc_decode(y_pred, input_length, greedy=True)[0][0].numpy()
    # i2w conversion
    y_pred = [[i2w[int(i)] for i in b if int(i) != -1] for b in y_pred]
    return y_pred

# --------------------

# Levenshtein distance between two sequences (a, b) at element level
def levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

# Compute the Symbol Error Rate (%) and the Sequence Error Rate (%) over a pair of ground-truth and predictions labels
# Labels are nested string lists with no padding
def compute_metrics(y_true: list, y_pred: list) -> Tuple[float, float]:
    ed_acc = 0
    length_acc = 0
    label_acc = 0
    counter = 0

    for t, h in zip(y_true, y_pred):
        ed = levenshtein(t, h)
        ed_acc += ed
        length_acc += len(t)
        if ed > 0:
            label_acc += 1
        counter += 1

    symer = 100. * ed_acc / length_acc
    seqer = 100. * label_acc / counter

    return symer, seqer

# --------------------

# Utility function for evaluting a model over a dataset and computing the corresponding metrics
def evaluate_model(model, omr_images_files, amt_images_files, labels_files, i2w):
    assert len(omr_images_files) == len(amt_images_files)
    y_pred_acc = []
    raw_y_pred_acc = []
    y_pred_acc = []
    y_pred_len_acc = []
    # Iterate over images in batches
    for start in range(0, len(omr_images_files), config.batch_size):
        omr_images, omr_images_len = list(zip(*[preprocess_image(i, config.omr_image_flag) for i in omr_images_files[start:start + config.batch_size]]))
        amt_images, amt_images_len = list(zip(*[preprocess_image(i, config.amt_image_flag) for i in amt_images_files[start:start + config.batch_size]]))
        # Zero-pad images to maximum batch image width
        max_width = max(omr_images + amt_images, key=np.shape).shape[1]
        omr_images = np.array([np.pad(i, pad_width=((0, 0), (0, max_width - i.shape[1]), (0, 0))) for i in omr_images], dtype="float32")
        amt_images = np.array([np.pad(i, pad_width=((0, 0), (0, max_width - i.shape[1]), (0, 0))) for i in amt_images], dtype="float32")
        images_len = [max(i_len, a_len) for i_len, a_len in zip(omr_images_len, amt_images_len)]
        # Obtain predictions
        xi = {"omr_image": omr_images, "amt_image": amt_images}
        y_pred = model(xi, training=False)
        # Append raw predictions and input lengths to accumulator variables to later save them
        raw_y_pred_acc.extend(y_pred.numpy())
        y_pred_len_acc.extend(images_len)
        # CTC greedy decoder (merge repeated, remove blanks, and i2w conversion)
        y_pred_acc.extend(ctc_greedy_decoder(y_pred, images_len, i2w))
    # Obtain true labels
    y_true_acc = [preprocess_label(i, training=False, w2i=None) for i in labels_files]
    # Compute metrics
    symer, seqer = compute_metrics(y_true_acc, y_pred_acc)
    print(f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples")
    return symer, seqer, [y_true_acc, raw_y_pred_acc, y_pred_len_acc]
