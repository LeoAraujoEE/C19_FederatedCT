import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from utils.dataset import Dataset
from utils.custom_generator import CustomDataGen, CustomDataGenerator

# Path to resized datasets in COVIDx CT-2A
path2datasets = os.path.join( ".", "data", "classification" )

# Builds object to handle radiopaedia dataset 
dataRadiopaedia = Dataset( import_dir = path2datasets, folder = "radiopaedia.org", 
                           input_col = "path_512", output_col = "class",  
                           keep_pneumonia = True, trainable = True )
dataRadiopaedia.load_dataframes()

# # Builds object to handle radiopaedia dataset
dataCOVID_CT_MD = Dataset( path2datasets, folder = "COVID-CT-MD", 
                          input_col = "path_256", output_col = "class", 
                          keep_pneumonia = True, trainable = True )
dataCOVID_CT_MD.load_dataframes()

hyperparameters = { "batchsize"          :                16, 
                    "input_size"         :     (512, 512, 3), 
                    "architecture"       :    "mobilenet_v2", 
                    "augmentation"       :             False, 
                    "preprocess_func"    :              True,
                  }

# List of data augmentation parameters
augmentation_dict = { "zoom":                    0.10,          # Max zoom in/zoom out
                      "shear":                   00.0,          # Max random shear
                      "rotation":                15.0,          # Max random rotation
                      "vertical_translation":    0.10,          # Max vertical translation
                      "horizontal_translation":  0.10,          # Max horizontal translation
                      "vertical_flip":          False,          # Allow vertical flips  
                      "horizontal_flip":        False,          # Allow horizontal flips    
                      "brightness":              0.00,          # Brightness adjustment range
                      "channel_shift":           00.0,          # Random adjustment to random channel
                      "fill_mode":          "constant"
                      }

generator = CustomDataGenerator( dataCOVID_CT_MD, "test", hyperparameters, aug_dict = augmentation_dict, 
                                 undersample = False, shuffle = False, seed = 42 )

fnames  = generator.get_fnames()
classes = generator.get_labels()

print("\n\n\n")
print(generator.class2label_dict)
for b, batch in enumerate(generator):

    if b == len(generator):
        break

    print( f"Batch {str(b+1).zfill(3)}/{len(generator)}, labels: {batch[1]}" )
        
    inputs, outputs = batch
    
    fig, axs = plt.subplots( nrows = 2, ncols = 8, figsize = (48, 36) )

    for i in range(inputs.shape[0]):
        
        img  = inputs[i]
        img  = (img + 1) / 2

        clss = outputs[i]
        print(f"clss: {clss}, clss.shape: {clss.shape}")
        lbel = generator.class2label_dict[clss[0]]

        r = i // 8
        c = i  % 8
        
        idx = b * 16 + i
        fname = fnames[idx]
        classe = classes[idx]
        print(idx, fname, classe)

        # 1° Coluna - Img Original
        axs[r][c].imshow(img, vmin=0, vmax=1, cmap = "gray")
        axs[r][c].set_title( "Label: {} - Class: {}/{}".format(lbel, clss, classe), size=11 )
        axs[r][c].set_xlabel( "File: '{}'\nShape: {}".format(fname, img.shape), size=11 )
        axs[r][c].grid(False)
        axs[r][c].set_xticks([])
        axs[r][c].set_yticks([])

    fig.tight_layout()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.close( fig )
print("\n\n\n")

# -------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------- Plots ---------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

n_rows = 2
n_cols = hyperparameters["batchsize"]
fig, axs = plt.subplots( nrows = n_rows, ncols = n_cols, figsize = (48, 36) )

print("\nPlotting images:")
for r, batch in enumerate(generator):

    if r == n_rows:
        break
        
    inputs, outputs = batch

    for c in range(n_cols):
        
        gray_img  = inputs[c]
        gray_img  = (gray_img + 1) / 2

        clss = outputs[c]
        lbel = generator.class2label_dict[clss]

        idx = r * n_cols + c
        fname = fnames[idx]
        print("File: '{}'".format(fname))

        # 1° Coluna - Img Original
        axs[r][c].imshow(gray_img, vmin=0, vmax=1, cmap = "gray")
        axs[r][c].set_title( "Label: {} - Class: {}".format(lbel, clss), size=14 )
        axs[r][c].set_xlabel( "File: '{}'\nShape: {}".format(fname, gray_img.shape), size=11 )
        axs[r][c].grid(False)
        axs[r][c].set_xticks([])
        axs[r][c].set_yticks([])

fig.tight_layout()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# -------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------- Plots ---------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------