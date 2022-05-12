# -*- coding: utf-8 -*-

import pathlib

# -- DATASET GLOBAL INFO -- #

# Camera-PrIMuS
# Calvo-Zaragoza, J.; Rizo, D. Camera-PrIMuS: Neural end-to-end Optical Music Recognition on realistic monophonic scores.
# In Proceedings of the 19th International Society for Music Information Retrieval Conference, Paris. 2018, pp. 248-255

base_path = "Corpus/End2End/Primus"
base_dir = pathlib.Path(base_path)
labels_dir = base_dir / "semantic"
folds_dir = base_dir / "5-crossval"
output_dir = base_dir / "EarlyFusion" / "Experiments"

label_extn = ".semantic"
# OMR
omr_image_extn = "_distorted.jpg"
omr_images_dir = base_dir / "jpg"
# cv2.IMREAD_COLOR
omr_image_flag = 1
# AMT
amt_image_extn = ".png"
amt_images_dir = base_dir / "cqt"
# cv2.IMREAD_UNCHANGED
amt_image_flag = -1

# -- ARCHITECTURE GLOBAL INFO -- #

# CNN either OMR-based

# OMR architecture fixed according to:
# Jorge Calvo-Zaragoza, Alejandro H. Toselli, Enrique Vidal
# Handwritten Music Recognition for Mensural notation with convolutional recurrent neural networks
# filters = [64, 64, 128, 128]
# kernel_size = [[5, 5], [5, 5], [3, 3], [3, 3]]
# pool_size = strides = [[2, 2], [2, 1], [2, 1], [2, 1]]
# leakyrelu_alpha = 0.2

# or AMT-based

# AMT architecture based on:
# Miguel A. Román, Antonio Pertusa, Jorge Calvo-Zaragoza
# Data representations for audio-to-score monophonic music transcription
# filters = [8, 8]
# kernel_size = [[10, 2], [8, 5]]
# pool_size = strides = [[2, 2], [2, 1]]
# leakyrelu_alpha = 0.2

# RNN
# lstm_units = [256, 256]
# lstm_dropout = 0.5

img_max_width = None
# This is true ONLY WHEN pool_size and strides have the same shape
width_reduction = 2

def set_arch_globals(cnn_type: str, batch=4):
    global img_max_height 
    global height_reduction
    global batch_size
    global arch_type
    arch_type = cnn_type
    batch_size = batch
    if arch_type == "omr":
        # OMR
        img_max_height = 64
        height_reduction = 16
    elif arch_type == "amt":
        # AMT
        img_max_height = 256
        height_reduction = 4

