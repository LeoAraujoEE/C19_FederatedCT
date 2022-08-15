import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from utils.custom_model_trainer import ModelManager

# 
PATH_DICT = { "datasets": os.path.join( "D:\\", "Datasets", "COVID19", "CT", "classification" ),
              "outputs" : os.path.join( "." ) }

# List of hyperparameter values
hyperparameter_dict = { "num_epochs":                     [5],  # Total N° of training epochs
                        "batchsize":                     [16],  # Minibatch size
                        "early_stop":                    [13],  # Early Stopping patience
                        "input_height":                 [256],  # Model's input size
                        "input_width":                  [256],  # Model's input size
                        "input_channels":                 [1],  # Model's input size
                        "start_lr":                    [1e-2],  # Starting learning rate
                        "lr_adjust_frac":               [0.5],  # Fraction to adjust learning rate
                        "lr_adjust_freq":                 [2],  # Frequency to adjust learning rate
                        "optimizer":                 ["adam"],  # Chosen optimizer
                        "monitor":                 ["val_f1"],  # Monitored variable for callbacks
                        "augmentation":                [True],  # If data augmentation should be used
                        "class_weights":              [False],  # If class_weights should be used
                        "apply_undersampling":         [True],  # Wether to apply Random Undersampling
                        "l1_reg":                         [0],  # Amount of L1 regularization
                        "l2_reg":                         [0],  # Amount of L2 regularization
                        "base_dropout":                 [.15],  # SpatialDropout2d between blocks in convolutional base
                        "top_dropout":                  [0.3],  # Dropout between dense layers in model top
                        "architecture":          ["resnet50"],  # Chosen architecture
                        "seed":                          [69],  # Seed for pseudorandom generators
                      } 

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
                      "constant_val":            00.0,
                      "fill_mode":          "constant"
                      }

keep_pneumonia = True
dataset_list = [ "miniCOVIDxCT", "Comp_CNCB_iCTCF", "miniCNCB", "radiopaedia.org", "COVID-CTSet", "COVID-CT-MD", "Comp_LIDC-SB" ]

trainManager = ModelManager( path_dict = PATH_DICT, dataset_name = "radiopaedia.org", 
                             hyperparam_values = hyperparameter_dict, 
                             aug_params = augmentation_dict, 
                             keep_pneumonia = keep_pneumonia )

trainManager.doGridSearch( shuffle = False )
