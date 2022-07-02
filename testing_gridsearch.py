import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

from utils.dataset import load_datasets
from utils.custom_model_trainer import ModelTrainer

# Path to resized datasets in COVIDx CT-3A
path2datasets = os.path.join( ".", "data", "classification" )

# Builds object to handle datasets for training and for external validation
dataTrain, dataVal_list = load_datasets( import_dir = path2datasets, 
                                         train_dataset = "radiopaedia.org", 
                                         input_col = "path_256", 
                                         output_col = "class", 
                                         keep_pneumonia = True )

# Parameters used to control SegModelTrainer
training_parameters = { "mock_steps":            5,             # Number of mock steps if a mock training is being performed
                        "mock_train":        False,             # If a mock training is being performed
                        "DEBUG_FLAG":        False,             # Flag to print debug info
                      }

# List of used models
model_list = [ "resnet_50v2", "densenet_121", "xception", "mobilenet_v2", "efficientnet_b0" ]

# List of hyperparameter values
hyperparameter_dict = { "num_epochs":                    [50],  # Total N° of training epochs
                        "batchsize":                     [16],  # Minibatch size
                        "early_stop":                    [13],  # Early Stopping patience
                        "input_size":         [(256, 256, 3)],  # Model's input size
                        "apply_undersampling":         [True],  # Wether to apply Random Undersampling
                        "start_lr":              [1e-3, 1e-4],  # Starting learning rate
                        "min_lr":                      [1e-4],  # Smallest learning rate value allowed
                        "lr_adjust_frac":              [0.70],  # N° of epochs between lr adjusts
                        "lr_patience":                    [4],  # N° of epochs between lr adjusts
                        "class_weights":              [False],  # If class_weights should be used
                        "preprocess_func":             [True],  # If keras preprocess_functions should be used
                        "monitor":                 ["val_f1"],  # Monitored variable for callbacks
                        "optimizer":                 ["adam"],  # Chosen optimizer
                        "l1_reg":                   [0, 1e-3],  # Amount of L1 regularization
                        "l2_reg":                   [0, 1e-3],  # Amount of L2 regularization
                        "dropout":                 [0.5, 0.0],  # Dropout for layers in skip connections
                        "augmentation":         [True, False],  # If data augmentation should be used
                        "pooling":                    ["avg"],  # Global Pooling used
                        "weights":                     [None],  # Pretrained weights
                        "architecture":            model_list,  # Chosen architecture
                        "seed":                          [69],  # Seed for pseudorandom generators
                      } 

augmentation_dict = { "zoom":                    0.10,          # Max zoom in/zoom out
                      "shear":                   00.0,          # Max random shear
                      "rotation":                15.0,          # Max random rotation
                      "vertical_translation":    0.10,          # Max vertical translation
                      "horizontal_translation":  0.10,          # Max horizontal translation
                      "vertical_flip":          False,          # Allow vertical flips  
                      "horizontal_flip":        False,          # Allow horizontal flips    
                      "brightness":              0.00,          # Brightness adjustment range
                      "channel_shift":           00.0,          # Random adjustment to random channel
                      "constant_val":            00.0,
                      "fill_mode":          "constant"
                      }

trainer = ModelTrainer( training_parameters, augmentation_dict, dataTrain, dataset_list = dataVal_list )
trainer.doGridSearch( hyperparameter_dict )