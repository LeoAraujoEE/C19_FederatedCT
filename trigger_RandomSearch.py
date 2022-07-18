import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from utils.custom_model_trainer import ModelManager

# List of used models
model_list = [ "resnet_50v2", "densenet_121", "xception", "mobilenet_v2", "efficientnet_b0" ]

# List of hyperparameter values
hyperparameter_ranges = { "num_epochs":                             (3,),  # Total N° of training epochs
                          "batchsize":                             (16,),  # Minibatch size
                          "early_stop":                            (13,),  # Early Stopping patience
                          "input_height":                         (256,),  # Model's input size
                          "input_width":                          (256,),  # Model's input size
                          "input_channels":                         (3,),  # Model's input size
                          "apply_undersampling":                 (True,),  # Wether to apply Random Undersampling
                          "start_lr":           ([1e-3, 1e-4], "sample"),  # Starting learning rate
                          "min_lr":                              (1e-5,),  # Smallest learning rate value allowed
                          "lr_adjust_frac":                      (0.70,),  # N° of epochs between lr adjusts
                          "lr_patience":                            (4,),  # N° of epochs between lr adjusts
                          "class_weights":                      (False,),  # If class_weights should be used
                          "monitor":                         ("val_f1",),  # Monitored variable for callbacks
                          "optimizer":                         ("adam",),  # Chosen optimizer
                          "l1_reg":                  (1e-4, 1e-2, "log"),  # Amount of L1 regularization
                          "l2_reg":                  (1e-4, 1e-2, "log"),  # Amount of L2 regularization
                          "dropout":                   (0.0, 0.5, "lin"),  # Dropout for layers in skip connections
                          "augmentation":                        (True,),  # If data augmentation should be used
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