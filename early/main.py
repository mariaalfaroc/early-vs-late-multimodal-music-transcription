# -*- coding: utf-8 -*-

import os

import tensorflow as tf

import config
from experimentation import k_fold_experiment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.list_physical_devices("GPU")

if __name__ == "__main__":
    epochs = 150
    arch_types = ["amt", "omr"]
    train_val_samples = [100, 500, 1000, 2000, None]
    for at in arch_types:
        for tv in train_val_samples:
            if tv is not None:
                train_samples = int(tv*0.8)
                val_samples = tv - train_samples
            else:
                train_samples = None
                val_samples = None
            config.set_arch_globals(cnn_type=at, batch=2)
            k_fold_experiment(epochs=epochs, num_train_samples=train_samples, num_val_samples=val_samples)
