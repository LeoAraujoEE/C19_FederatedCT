import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from utils.dataset import Dataset
from utils.custom_plots import CustomPlots
from utils.custom_generator import CustomDataGenerator

# Path to resized datasets in COVIDx CT-3A
path2datasets = os.path.join( "..", "..", "..", "..", "Datasets", "COVID19", "CT", "classification" )

# Builds object to handle radiopaedia dataset 
dataset = Dataset( import_dir = path2datasets, name = "COVID-CT-MD", 
                   keep_pneumonia = True )
dataset.load_dataframes()

hyperparameters = { "batchsize"          :                16, 
                    "input_height"       :               224, 
                    "input_width"        :               224, 
                    "input_channels"     :                 1, 
                    "architecture"       :    "mobilenet_v2", 
                    "augmentation"       :             False, 
                  }

# List of data augmentation parameters
augmentation_dict = { "zoom_in":                 0.00,          # Max zoom in
                      "zoom_out":                0.10,          # Max zoom out
                      "shear":                   00.0,          # Max random shear
                      "rotation":                 5.0,          # Max random rotation
                      "vertical_translation":    0.05,          # Max vertical translation
                      "horizontal_translation":  0.05,          # Max horizontal translation
                      "vertical_flip":          False,          # Allow vertical flips  
                      "horizontal_flip":        False,          # Allow horizontal flips    
                      "brightness":              0.00,          # Brightness adjustment range
                      "channel_shift":           00.0,          # Random adjustment to random channel
                      "constant_val":            00.0,          # Random adjustment to random channel
                      "fill_mode":          "constant"
                      }

generator = CustomDataGenerator( dataset, "test", hyperparameters, aug_dict = augmentation_dict, 
                                 sampling = "oversampling", shuffle = True, seed = 42 )

for b in range(len(generator)):
    fig = CustomPlots.plot_batch(generator, batch = b, n_cols = 8, figsize = (16, 9))
    plt.close(fig)