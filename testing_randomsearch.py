import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

from utils.dataset import Dataset
from utils.custom_model_trainer import SegModelTrainer

# Path to Oxford Pets dataset
path2dataset = "/home/leo/Datasets"
import_dir   = os.path.join( path2dataset, "Oxford Pets" )

# Builds object to handle Oxford Pets dataset
dataPets = Dataset( import_dir, trainable = True )

# Creates copies of dataPets to mock cross-validation
cval_list = [ Dataset( os.path.join( path2dataset, "Oxford Petz" ), trainable = False ) for _ in range(1) ]

# Parameters used to control SegModelTrainer
training_parameters = { "mock_steps":           10,          # Number of mock steps if a mock training is being performed
                        "mock_train":        False,          # If a mock training is being performed
                        "DEBUG_FLAG":        False,          # Flag to print debug info
                      }

# List of hyperparameter values
hyperparameter_ranges = { "num_epochs":                             (30,),       # Total number of training epochs
                          "batchsize":                 ([2, 8], "sample"),       # Minibatch size
                          "early_stop":                             (15,),       # Early Stopping patience
                          "input_size":                  ((128, 128, 3),),       # Model's input size
                          "start_lr":                             (1e-4,),       # Starting learning rate
                          "final_lr":                             (1e-5,),       # Final value for learning rate
                          "lr_adjust_freq":                          (5,),       # Number of epochs between lr adjusts
                          "monitor":                         ("val_IoU",),       # Monitored variable for callbacks
                          "optimizer":                          ("adam",),       # Chosen optimizer
                          "l1_reg":                   (1e-6, 1e-3, "log"),       # Amount of L1 regularization
                          "l2_reg":                   (1e-6, 1e-3, "log"),       # Amount of L2 regularization
                          "skip_dropout":             (1e-6, 3e-1, "log"),       # Dropout for layers in skip connections
                          "neck_dropout":             (0.30, 0.75, "log"),       # Dropout fpr layers in the bottleneck
                          "num_blocks":                              (3,),       # Number of stages in U-Net/ResU-Net
                          "architecture": (["unet", "resunet"], "sample"),       # Chosen architecture
                          "seed":                                   (69,),       # Seed for pseudorandom generators
                        } 

# List of data augmentation parameters
d_augmentation_dict = { "zoom":                    0.15,     # Max zoom in/zoom out
                        "shear":                   30.0,     # Max random shear
                        "rotation":                45.0,     # Max random rotation
                        "vertical_translation":    0.20,     # Max vertical translation
                        "horizontal_translation":  0.20,     # Max horizontal translation
                        "vertical_flip":           True,     # Allow vertical flips  
                        "horizontal_flip":         True,     # Allow horizontal flips    
                        "brightness":              0.00,     # Brightness adjustment range
                        "channel_shift":           30.0,     # Random adjustment to random channel
                        "grayscale_prob":          0.25,     # Probability to convert image to grayscale
                        "shuffle_rgb_prob":        0.10,     # Probability to shuffle image's rgb channels
                        "sp_noise_prob":           0.30,     # Probability of adding salt & pepper noise
                        "gaussian_noise_prob":     0.30,     # Probability of adding gaussian noise
                        "grid_mask_prob":          0.00,     # Probability of applying GridMask
                        "grid_mask_hole_size":     0.60,     # Hole fraction of each unit in GridMask
                        "grid_mask_unit_size":     0.30,     # Average size of the unit in GridMask
                        "hide_seek_prob":          0.00,     # Probability of applying Hide&Seek
                        "hide_seek_dropchance":    0.25,     # Probability of dropping a unit in Hide&Seek
                        "hide_seek_unit_size":     0.25      # Average size of units in Hide&Seek
                      }

trainer = SegModelTrainer( training_parameters, d_augmentation_dict, dataPets)#, dataset_list = cval_list )

trainer.doRandomSearch( hyperparameter_ranges, n_models = 25 )