import os
from utils.custom_model_trainer import ModelManager

# 
PATH_DICT = { "datasets": os.path.join( "D:\\", "Datasets", "COVID19", "CT", "classification" ),
              "outputs" : os.path.join( "." ) }

# List of hyperparameter values
hyperparameter_ranges = { "num_epochs":                             (3,),  # Total N° of training epochs
                          "batchsize":                             (32,),  # Minibatch size
                          "early_stop":                            (11,),  # Early Stopping patience
                          "input_height":                         (224,),  # Model's input size
                          "input_width":                          (224,),  # Model's input size
                          "input_channels":                         (1,),  # Model's input size
                          "start_lr":           ([1e-3, 1e-4], "sample"),  # Starting learning rate
                          "lr_adjust_frac":                       (0.5,),  # Fraction to adjust learning rate
                          "lr_adjust_freq":                         (5,),  # Frequency to adjust learning rate
                          "optimizer":                         ("adam",),  # Chosen optimizer
                          "monitor":                         ("val_f1",),  # Monitored variable for callbacks
                          "augmentation":                        (True,),  # If data augmentation should be used
                          "class_weights":                      (False,),  # If class_weights should be used
                          "apply_undersampling":                 (True,),  # Wether to apply Random Undersampling
                          "l1_reg":                  (1e-4, 1e-2, "log"),  # Amount of L1 regularization
                          "l2_reg":                  (1e-4, 1e-2, "log"),  # Amount of L2 regularization
                          "base_dropout":                         (0.3,),  # Dropout for layers in skip connections
                          "top_dropout":                          (0.5,),  # Dropout for layers in skip connections
                          "architecture":                  ("resnet50",),  # Chosen architecture
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
  
trainManager = ModelManager( path_dict = PATH_DICT, 
                             dataset_name = "radiopaedia.org", 
                             hyperparam_values = hyperparameter_ranges, 
                             aug_params = augmentation_dict, 
                             keep_pneumonia = False )

trainManager.doRandomSearch( n_models = 1 )