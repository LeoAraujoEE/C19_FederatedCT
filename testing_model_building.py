import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params

from utils.custom_models import ModelBuilder

def show_attr( obj1, obj2, n_layer, show = "all" ):
  if not show in ["all", "equal", "diff"]:
    show = "all"
    print("Wrong value for show! Try one of these: ['all', 'equal', 'diff']...")
  
  prefix = f""
  objs = [obj1, obj2]
  if all([isinstance(obj, tf.keras.layers.MaxPooling2D) for obj in objs]):
    has_activation = False
    attr_list = ["pool_size", "strides", "padding"]
    
  elif all([isinstance(obj, tf.keras.layers.Conv2D) for obj in objs]):
    if not all([("_0_conv" in obj.name.lower()) or ("_num0" in obj.name.lower()) for obj in objs]):
      n_layer += 1
      prefix = f"{str(n_layer).zfill(3)} "
    has_activation = True
    attr_list = ["filters", "kernel_size", "strides", "padding", "use_bias"]
    
  elif all([isinstance(obj, tf.keras.layers.Dense) for obj in objs]):
    n_layer += 1
    prefix = f"{str(n_layer).zfill(3)} "
    has_activation = True
    attr_list = ["units", "use_bias"]
    
  print(f"\n\t{prefix}Name: {obj1.name} x {obj2.name}:")
  
  if has_activation:
    actv1, actv2 = str(obj1.activation).split(" ")[1], str(obj2.activation).split(" ")[1]
    is_equal = (actv1 == actv2)
    if (show == "all") or (show == "equal" and is_equal) or (show == "diff" and not is_equal):
      print(f"\t\tActivations: {actv1} x {actv2}")
    
    parm1, parm2 = count_params(obj1.trainable_weights), count_params(obj2.trainable_weights)
    is_equal = (parm1 == parm2)
    if (show == "all") or (show == "equal" and is_equal) or (show == "diff" and not is_equal):
      print(f"\t\tParameters: {parm1:,} x {parm2:,}")
  
  for attrib in attr_list:
    is_equal = getattr(obj1, attrib) == getattr(obj2, attrib)
    if (show == "all") or (show == "equal" and is_equal) or (show == "diff" and not is_equal):
      print(f"\t\t{attrib.title()}: {is_equal} - {getattr(layer_k, attrib)} x {getattr(layer_m, attrib)}")
    
  return n_layer

# vgg_16, vgg_19,  
# mobilenet_v2, xception, densenet_121, densenet_169, densenet_201, 
# efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, 
# efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

# List of hyperparameter possible values
hyperparameter_dict = { "num_epochs":                      50,  # Total number of training epochs
                        "batchsize":                        8,  # Minibatch size
                        "input_height":                   299,  # Model's input height
                        "input_width":                    299,  # Model's input width
                        "input_channels":                   1,  # Model's input channels
                        "start_lr":                      1e-3,  # Starting learning rate
                        "min_lr":                        1e-5,  # Smallest learning rate value allowed
                        "monitor":                   "val_f1",  # Monitored variable for callbacks
                        "optimizer":                   "adam",  # Chosen optimizer
                        "l1_reg":                        1e-3,  # Amount of L1 regularization
                        "l2_reg":                        1e-4,  # Amount of L2 regularization
                        "base_dropout":                     0,  # Dropout between dense layers
                        "top_dropout":                      0,  # Dropout between dense layers
                        "pooling":                      "avg",  # Global Pooling used
                        "weights":                       None,  # Pretrained weights
                        "architecture":  "custom_inceptionv4",  # Chosen architecture
                      }       

model_dir  = os.path.join( ".", "output", "models", "joao_123" )
model_path = os.path.join( model_dir, f"{hyperparameter_dict['architecture']}.h5" )

model_builder = ModelBuilder( model_path = model_path, gen_fig = True )
model_mine = model_builder( hyperparameter_dict, seed = 69 )

# hyperparameter_dict["architecture"] = "densenet_201"
# model_keras = model_builder( hyperparameter_dict, seed = 69 )

input_size = (hyperparameter_dict["input_height"], hyperparameter_dict["input_width"], hyperparameter_dict["input_channels"])
model_keras = tf.keras.applications.InceptionV3( input_shape = input_size, include_top = True, classes = 1, weights = None )
path = os.path.join (".", "output", "models", "joao_123", "inceptionv3_keras.png" )
tf.keras.utils.plot_model( model_keras, to_file = path, show_shapes = True, show_layer_names = True, 
                            rankdir = "TB", expand_nested = False, dpi = 96 )
        
# Counts the model's parameters
trainable_count = int(np.sum([ count_params(l.trainable_weights) for l in model_keras.layers ]))
non_trainable_count = int(np.sum([ count_params(l.non_trainable_weights) for l in model_keras.layers ]))
print("\nCreated model with {:,} trainable parameters and {:,} non trainable ones...".format(trainable_count, non_trainable_count))

k_layer_idxs = [ i for i, layer in enumerate(model_keras.layers) if isinstance(layer, tf.keras.layers.MaxPooling2D) or isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense) ]
m_layer_idxs = [ i for i, layer in enumerate(model_mine.layers) if isinstance(layer, tf.keras.layers.MaxPooling2D) or isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense) ]
assert len(k_layer_idxs) == len(m_layer_idxs)

n_layers = 0
for idx_k, idx_m in zip(k_layer_idxs, m_layer_idxs):
  layer_m = model_mine.layers[idx_m]
  layer_k = model_keras.layers[idx_k]
  
  n_layers = show_attr( layer_k, layer_m, n_layers, show = "diff" )

k_other_layers = [ layer for i, layer in enumerate(model_keras.layers) if not i in k_layer_idxs ]
trainable_count = int(np.sum([ count_params(l.trainable_weights) for l in k_other_layers ]))
non_trainable_count = int(np.sum([ count_params(l.non_trainable_weights) for l in k_other_layers ]))
print("\nKeras Other Layers have {:,} trainable parameters and {:,} non trainable ones...".format(trainable_count, non_trainable_count))

m_other_layers = [ layer for i, layer in enumerate(model_mine.layers) if not i in m_layer_idxs ]
trainable_count = int(np.sum([ count_params(l.trainable_weights) for l in m_other_layers ]))
non_trainable_count = int(np.sum([ count_params(l.non_trainable_weights) for l in m_other_layers ]))
print("\nMine Other Layers have {:,} trainable parameters and {:,} non trainable ones...".format(trainable_count, non_trainable_count))