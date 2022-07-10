import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from utils.dataset import Dataset
from utils.custom_plots import CustomPlots
from utils.custom_generator import CustomDataGenerator

# Path to resized datasets in COVIDx CT-3A
path2datasets = os.path.join( ".", "data", "classification" )

# Builds object to handle radiopaedia dataset 
dataset = Dataset( import_dir = path2datasets, folder = "radiopaedia.org", 
                           input_col = "path_512", output_col = "class",  
                           keep_pneumonia = True, trainable = True )
dataset.load_dataframes()

hyperparameters = { "batchsize"          :                16, 
                    "input_height"       :               512, 
                    "input_width"        :               512, 
                    "input_channels"     :                 3, 
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

generator = CustomDataGenerator( dataset, "test", hyperparameters, aug_dict = augmentation_dict, 
                                 undersample = True, shuffle = False, seed = 42 )

for b in range(len(generator)):
    fig = CustomPlots.plot_batch(generator, batch = b, n_cols = 8, figsize = (16, 9))
    plt.close(fig)