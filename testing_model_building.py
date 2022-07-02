import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

from utils.custom_models import ModelBuilder


            # vgg_16, vgg_19, resnet_50v2, resnet_101v2, resnet_152v2, 
            # mobilenet_v2, xception, densenet_121, densenet_169, densenet_201, 
            # efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, 
            # efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

# List of hyperparameter possible values
hyperparameter_dict = { "num_epochs":                      50,  # Total number of training epochs
                        "batchsize":                        8,  # Minibatch size
                        "input_size":           (224, 224, 3),  # Model's input size
                        "start_lr":                      1e-3,  # Starting learning rate
                        "min_lr":                        1e-5,  # Smallest learning rate value allowed
                        "monitor":                  "val_IoU",  # Monitored variable for callbacks
                        "optimizer":                   "adam",  # Chosen optimizer
                        "l1_reg":                        1e-3,  # Amount of L1 regularization
                        "l2_reg":                        1e-4,  # Amount of L2 regularization
                        "dropout":                          0,  # Dropout between dense layers
                        "pooling":                      "max",  # Global Pooling used
                        "weights":                       None,  # Pretrained weights
                        "architecture":        "efficientnet_b1",  # Chosen architecture
                      }       

model_dir  = os.path.join( ".", "output", "models", "joao_123" )
model_path = os.path.join( model_dir, "joao_123.h5" )

model_builder = ModelBuilder( model_path = model_path )
# model_builder = ModelBuilder( model_path = model_path, gen_fig = True)
model_a = model_builder( hyperparameter_dict, n_classes = 39, seed = 69 )
# model_a = model_builder.unfreeze_blocks( model_a, 1 )
# model_b = model_builder( hyperparameter_dict, n_classes = 39, seed = 69 )

# # print( model.summary() )
# import numpy as np

# for layer_a, layer_b in zip(model_a.layers, model_b.layers):
#     Wa = layer_a.get_weights()
#     Wb = layer_b.get_weights()
#     print( "\t",layer_a.name, [ (wa == wb).all() for idx, (wa, wb) in enumerate(zip(Wa, Wb))] )

# model_a.summary()