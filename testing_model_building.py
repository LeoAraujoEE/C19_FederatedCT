import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from utils.custom_models import ModelBuilder

# vgg_16, vgg_19, resnet_50v2, resnet_101v2, resnet_152v2, 
# mobilenet_v2, xception, densenet_121, densenet_169, densenet_201, 
# efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, 
# efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

# List of hyperparameter possible values
hyperparameter_dict = { "num_epochs":                      50,  # Total number of training epochs
                        "batchsize":                        8,  # Minibatch size
                        "input_height":                   224,  # Model's input height
                        "input_width":                    224,  # Model's input width
                        "input_channels":                   1,  # Model's input channels
                        "start_lr":                      1e-3,  # Starting learning rate
                        "min_lr":                        1e-5,  # Smallest learning rate value allowed
                        "monitor":                  "val_IoU",  # Monitored variable for callbacks
                        "optimizer":                   "adam",  # Chosen optimizer
                        "l1_reg":                        1e-3,  # Amount of L1 regularization
                        "l2_reg":                        1e-4,  # Amount of L2 regularization
                        "base_dropout":                     0,  # Dropout between dense layers
                        "top_dropout":                      0,  # Dropout between dense layers
                        "pooling":                      "max",  # Global Pooling used
                        "weights":                       None,  # Pretrained weights
                        "architecture":           "resnet_50v2",  # Chosen architecture
                      }       

model_dir  = os.path.join( ".", "output", "models", "joao_123" )
model_path = os.path.join( model_dir, "joao_123.h5" )

model_builder = ModelBuilder( model_path = model_path, gen_fig = True )
model_keras = model_builder( hyperparameter_dict, seed = 69 )

hyperparameter_dict["architecture"] = "custom_resnet34"
model_mine = model_builder( hyperparameter_dict, seed = 69 )

# print( model_a.summary() )

import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params

print("\t".join(["num", "name", "filters", "size", "strides", "padding", "activation"]))

k_layer_idxs = [ i for i, layer in enumerate(model_keras.layers) if isinstance(layer, tf.keras.layers.MaxPooling2D) or isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense) ]
m_layer_idxs = [ i for i, layer in enumerate(model_mine.layers) if isinstance(layer, tf.keras.layers.MaxPooling2D) or isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense) ]
assert len(k_layer_idxs) == len(m_layer_idxs)

n_layers = 0
for idx_k, idx_m in zip(k_layer_idxs, m_layer_idxs):
  layer_m = model_mine.layers[idx_m]
  layer_k = model_keras.layers[idx_k]
  
  if isinstance(layer_k, tf.keras.layers.MaxPooling2D) and isinstance(layer_m, tf.keras.layers.MaxPooling2D):
    print(f"\n\tName: {layer_k.name} x {layer_m.name}:")
    for attrib in ["pool_size", "strides", "padding"]:
      if getattr(layer_k, attrib) != getattr(layer_m, attrib):
          print(f"\t\t{attrib.title()}: {getattr(layer_k, attrib) == getattr(layer_m, attrib)} - {getattr(layer_k, attrib)} x {getattr(layer_m, attrib)}")
  
  if isinstance(layer_k, tf.keras.layers.Conv2D) and isinstance(layer_m, tf.keras.layers.Conv2D):
    
    if ("_0_conv" in layer_k.name.lower()) and ("_num0" in layer_m.name.lower()):
      print(f"\n\t{str(n_layers).zfill(2)} Name: {layer_k.name} x {layer_m.name}:")
      for attrib in ["filters", "kernel_size", "strides", "padding", "params", "use_bias"]:
        if attrib == "params":
          print(f"\t\tParameters: {count_params(layer_k.trainable_weights):,} x {count_params(layer_m.trainable_weights):,}")
        else:
          print(f"\t\t{attrib.title()}: {getattr(layer_k, attrib) == getattr(layer_m, attrib)} - {getattr(layer_k, attrib)} x {getattr(layer_m, attrib)}")
      continue
    
    n_layers+=1
    print(f"\n{str(n_layers).zfill(2)} Name: {layer_k.name}x{layer_m.name}:")
    for attrib in ["name", "filters", "kernel_size", "strides", "padding", "params", "use_bias"]:
      if attrib == "params":
        print(f"\tParameters: {count_params(layer_k.trainable_weights):,} x {count_params(layer_m.trainable_weights):,}")
      else:
        if getattr(layer_k, attrib) != getattr(layer_m, attrib):
          print(f"\t{attrib.title()}: {getattr(layer_k, attrib) == getattr(layer_m, attrib)} - {getattr(layer_k, attrib)} x {getattr(layer_m, attrib)}")
  
  if isinstance(layer_k, tf.keras.layers.Dense) and isinstance(layer_m, tf.keras.layers.Dense):
    n_layers+=1
    print(f"\n{str(n_layers).zfill(2)} Name: {layer_k.name} x {layer_m.name}:")
    for attrib in ["units", "params", "use_bias"]:
      if attrib == "params":
        print(f"\tParameters: {count_params(layer_k.trainable_weights):,} x {count_params(layer_m.trainable_weights):,}")
      else:
        if getattr(layer_k, attrib) != getattr(layer_m, attrib):
          print(f"\t{attrib.title()}: {getattr(layer_k, attrib) == getattr(layer_m, attrib)} - {getattr(layer_k, attrib)} x {getattr(layer_m, attrib)}")