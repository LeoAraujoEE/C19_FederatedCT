import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from utils.custom_model_trainer import ModelManager

# List of used models
model_list = [ "resnet_50v2", "densenet_121", "xception", "mobilenet_v2", "efficientnet_b0" ]

# List of hyperparameter values
hyperparameter_ranges = { "num_epochs":                            (30,),  # Total N° of training epochs
                          "batchsize":                             (16,),  # Minibatch size
                          "early_stop":                            (13,),  # Early Stopping patience
                          "input_height":                         (256,),  # Model's input size
                          "input_width":                          (256,),  # Model's input size
                          "input_channels":                         (1,),  # Model's input size
                          "start_lr":           ([1e-3, 1e-4], "sample"),  # Starting learning rate
                          "lr_adjust_frac":                       (0.1,),  # Fraction to adjust learning rate
                          "lr_adjust_freq":                        (10,),  # Frequency to adjust learning rate
                          "optimizer":                         ("adam",),  # Chosen optimizer
                          "monitor":                         ("val_f1",),  # Monitored variable for callbacks
                          "augmentation":                        (True,),  # If data augmentation should be used
                          "class_weights":                      (False,),  # If class_weights should be used
                          "apply_undersampling":                 (True,),  # Wether to apply Random Undersampling
                          "l1_reg":                  (1e-4, 1e-2, "log"),  # Amount of L1 regularization
                          "l2_reg":                  (1e-4, 1e-2, "log"),  # Amount of L2 regularization
                          "base_dropout":            (0.05, 0.25, "lin"),  # Dropout for layers in skip connections
                          "top_dropout":             (0.25, 0.50, "lin"),  # Dropout for layers in skip connections
                          "architecture":         (model_list, "sample"),  # Chosen architecture
                          "seed":                                  (69,),  # Seed for pseudorandom generators
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

trainManager = ModelManager( "radiopaedia.org", hyperparameter_ranges, augmentation_dict, keep_pneumonia = True )
trainManager.doRandomSearch( hyperparameter_ranges, n_models = 3 )