# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import layers

import config

def ctc_loss_lambda(args):
    y_true, y_pred, input_length, label_length = args
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def amt_cnn(task: str):
    return keras.Sequential([
        layers.Conv2D(8, (10, 2), padding="same", use_bias=False, name=f"Conv2D_1_{task}"),
        layers.BatchNormalization(name=f"BatchNorm_1_{task}"),
        layers.LeakyReLU(0.2, name=f"LeakyReLU_1_{task}"),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=f"MaxPool2D_1_{task}"),
        layers.Conv2D(8, (8, 5), padding="same", use_bias=False, name=f"Conv2D_2_{task}"),
        layers.BatchNormalization(name=f"BatchNorm_2_{task}"),
        layers.LeakyReLU(0.2, name=f"LeakyReLU_2_{task}"),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), name=f"MaxPool2D_2_{task}")
    ])

def omr_cnn(task: str):
    return keras.Sequential([
        layers.Conv2D(64, 5, padding="same", use_bias=False, name=f"Conv2D_1_{task}"),
        layers.BatchNormalization(name=f"BatchNorm_1_{task}"),
        layers.LeakyReLU(0.2, name=f"LeakyReLU_1_{task}"),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=f"MaxPool2D_1_{task}"),
        layers.Conv2D(64, 5, padding="same", use_bias=False, name=f"Conv2D_2_{task}"),
        layers.BatchNormalization(name=f"BatchNorm_2_{task}"),
        layers.LeakyReLU(0.2, name=f"LeakyReLU_2_{task}"),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), name=f"MaxPool2D_2_{task}"),
        layers.Conv2D(128, 3, padding="same", use_bias=False, name=f"Conv2D_3_{task}"),
        layers.BatchNormalization(name=f"BatchNorm_3_{task}"),
        layers.LeakyReLU(0.2, name=f"LeakyReLU_3_{task}"),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), name=f"MaxPool2D_3_{task}"),
        layers.Conv2D(128, 3, padding="same", use_bias=False, name=f"Conv2D_4_{task}"),
        layers.BatchNormalization(name=f"BatchNorm_4_{task}"),
        layers.LeakyReLU(0.2, name=f"LeakyReLU_4_{task}"),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), name=f"MaxPool2D_4_{task}")
    ])

def get_cnn(arch_type: str, task: str):
    if arch_type == "amt":
        return amt_cnn(task=task)
    return omr_cnn(task=task)

def build_models(num_labels: int):
    # Input block
    omr_image = keras.Input(shape=(config.img_max_height, config.img_max_width, 1), dtype="float32", name="omr_image")
    amt_image = keras.Input(shape=(config.img_max_height, config.img_max_width, 1), dtype="float32", name="amt_image")
    image_len = keras.Input(shape=(1,), dtype="int32", name="image_len")
    label = keras.Input(shape=(None,), dtype="int32", name="label")
    label_len = keras.Input(shape=(1,), dtype="int32", name="label_len")

    # Convolutional block
    # This is where they differ: we have speficic CNN models for each type of data
    # Right now, they are the same in terms of configuration
    # OMR convolutional block
    omr_cnn = get_cnn(arch_type=config.arch_type, task="omr")
    omr_x = omr_cnn(omr_image)

    # AMT convolutional block
    amt_cnn = get_cnn(arch_type=config.arch_type, task="amt")
    amt_x = amt_cnn(amt_image)

    # Merge features
    x = layers.Average(name="Average")([omr_x, amt_x])

    # Intermediate block (preparation to enter the recurrent one)
    # [batch, height, width, channels] -> [batch, width, height, channels] 
    x = layers.Permute((2, 1, 3), name="Permute")(x)
    # [batch, width, height, channels] -> [batch, width, height * channels]
    x = layers.Reshape((-1, x.shape[2] * x.shape[3]), name="Reshape")(x)

    # Recurrent block
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5, name="LSTM_1"), name="Bidirectional_1")(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5, name="LSTM_2"), name="Bidirectional_2")(x)

    # Dense layer
    # num_classes -> represents "num_labels + 1" classes, where num_labels is the number of true labels, 
    # and the largest value  "(num_classes - 1) = (num_labels + 1 - 1) = num_labels is reserved for the blank label
    # Range of true labels -> [0, len(voc_size))
    # Therefore, len(voc_size) is the default value for the CTC-blank index
    output = layers.Dense(num_labels + 1, activation="softmax", name="Dense")(x)

    # CTC-loss computation
    # Keras does not currently support loss functions with extra parameters, so CTC loss is implemented in a Lambda layer
    ctc_loss = layers.Lambda(
        function=ctc_loss_lambda,
        output_shape=(1,),
        name="ctc_loss" 
    )([label, output, image_len, label_len])

    # Create training model and predicition model
    model = keras.Model([omr_image, amt_image, image_len, label, label_len], ctc_loss)
    # The loss calculation is already done, so use a dummy lambda function for the loss
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={"ctc_loss" : lambda y_true, y_pred: y_pred}
    )
    # At inference time, we only have the images as inputs and the softmax prediction as output
    prediction_model = keras.Model([model.get_layer("omr_image").input, model.get_layer("amt_image").input], output)

    print(model.summary())

    return model, prediction_model
