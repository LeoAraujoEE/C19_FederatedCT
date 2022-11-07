import os
from utils.custom_model_trainer import ModelManager

KEEP_PNEUMONIA = True
SUBDIR = "remapped" if KEEP_PNEUMONIA else "dropped"

# 
PATH_DICT = { "datasets": os.path.join( "..", "data", "Processed", "CT", "classification", "COVIDxCT-3A" ),
              "outputs" : os.path.join( "..", "output", "mock_fl_models", SUBDIR ), 
              # "outputs" : os.path.join( "..", "output", "fl_models", SUBDIR ) 
            }

# List of parameters for Federated Learning simulation
fedlearn_params = { "epochs_per_step":                [2],
                    "max_steps_frac" :             [0.00],
                    "client_frac"    :             [1.00],
                    "aggregation"    :            ["avg"],
                  }

# List of hyperparameter values
hyperparameters = { "num_epochs":                     [5],  # Total N° of training epochs
                    "batchsize":                     [32],  # Minibatch size
                    "early_stop_patience":            [1],  # Early Stopping patience
                    "early_stop_delta":           [0.001],  # Minimum improvement for early stopping
                    "input_height":                 [224],  # Model's input size
                    "input_width":                  [224],  # Model's input size
                    "input_channels":                 [1],  # Model's input size
                    "start_lr":                    [1e-2],  # Starting learning rate
                    "lr_adjust_frac":              [0.90],  # Fraction to adjust learning rate
                    "lr_adjust_freq":                 [2],  # Frequency to adjust learning rate
                    "optimizer":                 ["adam"],  # Chosen optimizer
                    "monitor":                 ["val_f1"],  # Monitored variable for callbacks
                    "augmentation":                [True],  # If data augmentation should be used
                    "class_weights":              [False],  # If class_weights should be used
                    "sampling":          ["oversampling"],  # Chosen sampling method (None, over/under sampling)
                    "l1_reg":                      [1e-5],  # Amount of L1 regularization
                    "l2_reg":                      [1e-5],  # Amount of L2 regularization
                    "base_dropout":                 [0.2],  # SpatialDropout2d between blocks in convolutional base
                    "top_dropout":                  [0.3],  # Dropout between dense layers in model top
                    "architecture":          ["resnet18"],  # Chosen architecture
                    "seed":                          [69],  # Seed for pseudorandom generators
                  } 

data_aug_params = { "zoom_in":                     0.00,  # Max zoom in
                    "zoom_out":                    0.10,  # Max zoom out
                    "shear":                       00.0,  # Max random shear
                    "rotation":                     5.0,  # Max random rotation
                    "vertical_translation":        0.05,  # Max vertical translation
                    "horizontal_translation":      0.05,  # Max horizontal translation
                    "vertical_flip":              False,  # Allow vertical flips  
                    "horizontal_flip":            False,  # Allow horizontal flips    
                    "brightness":                  0.00,  # Brightness adjustment range
                    "channel_shift":               00.0,  # Random adjustment to random channel
                    "constant_val":                00.0,
                    "fill_mode":              "constant"
                    }

dataset_list = [ "COVIDxCT",        # Whole COVIDxCT-3A dataset
                 "miniCOVIDxCT",    # Reduced COVIDxCT-3A, has only samples from used datasets
                 "Comp_CNCB_iCTCF", # 88k / 69k - Combination of CNCB non COVID samples + iCTCF
                 "miniCNCB",        # 74k / 55k - Rest of CNCB dataset
                 "COVID-CT-MD",     # 23k / 20k - 
                 "Comp_LIDC-SB",    # 18k / 18k - Combination of LIDC + Stone Brook
                 "COVID-CTSet",     # 12k / 12k - 
                 "radiopaedia.org", #  4k /  3k
               ]
  
trainManager = ModelManager( path_dict = PATH_DICT, 
                             dataset_name = "miniCOVIDxCT", 
                             fl_params = fedlearn_params,
                             hyperparam_values = hyperparameters, 
                             aug_params = data_aug_params, 
                             keep_pneumonia = KEEP_PNEUMONIA )

trainManager.doGridSearch( shuffle = False )