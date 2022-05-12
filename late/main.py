# -*- coding: utf-8 -*-

import os

import tensorflow as tf

import config
from experimentation import k_fold_experiment

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.list_physical_devices("GPU")

if __name__ == "__main__":
    epochs = 150
    train_val_samples = [100, 500, 1000, 2000, None]
    for tv in train_val_samples:
        if tv is not None:
            train_samples = int(tv*0.8)
            val_samples = tv - train_samples
        else:
            train_samples = None
            val_samples = None
        # OMR
        config.set_task(value="omr")
        config.set_data_globals()
        config.set_arch_globals(batch=16)
        print(f"Task == {config.task}")
        k_fold_experiment(epochs=epochs, num_train_samples=train_samples, num_val_samples=val_samples)
        # AMT
        config.set_task(value="amt")
        config.set_data_globals()
        config.set_arch_globals(batch=4)
        print(f"Task == {config.task}")
        k_fold_experiment(epochs=epochs, num_train_samples=train_samples, num_val_samples=val_samples)
