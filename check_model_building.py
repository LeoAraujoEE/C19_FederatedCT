import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params

from utils.custom_models import ModelBuilder

def has_params(obj):
  if hasattr(obj, "trainable_weights"):
    if int(count_params(obj.trainable_weights)) > 0:
      return True
    
  if hasattr(obj, "non_trainable_weights"):
    if int(count_params(obj.non_trainable_weights)) > 0:
      return True
    
  return False

def is_conv(obj):
  return any([ isinstance(obj, tf.keras.layers.Conv2D),
               isinstance(obj, tf.keras.layers.SeparableConv2D),
               isinstance(obj, tf.keras.layers.DepthwiseConv2D) ])
  
def show_attr( obj1, obj2, n_layer, show = "all" ):
  if not show in ["all", "equal", "diff"]:
    show = "all"
    print("Wrong value for show! Try one of these: ['all', 'equal', 'diff']...")
  
  prefix = f""
  objs = [obj1, obj2]
  if all([isinstance(obj, tf.keras.layers.MaxPooling2D) for obj in objs]):
    has_activation = False
    attr_list = ["pool_size", "strides", "padding"]
    
  elif all([is_conv(obj) for obj in objs]):
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
      print(f"\t\t{attrib.title()}: {is_equal} - {getattr(obj1, attrib)} x {getattr(obj2, attrib)}")
    
  return n_layer

def remove_normalization_layers(model, start = 3):
    # Used to remove normalization layers from keras' EfficientNetV1
    # Based on: https://stackoverflow.com/questions/67176547/how-to-remove-first-n-layers-from-a-keras-model
    confs = model.get_config()
    kept_layers = set()
    
    for i, l in enumerate(confs['layers']):
        if i == 0:
            confs['layers'][0]['config']['batch_input_shape'] = model.layers[start].input_shape
            if i != start:
                #confs['layers'][0]['name'] += str(random.randint(0, 100000000)) # rename the input layer to avoid conflicts on merge
                confs['layers'][0]['config']['name'] = confs['layers'][0]['name']
                
        elif i < start:
            continue
          
        kept_layers.add(l['name'])
        if l['class_name'] == "Dense":
          break
        
    # filter layers
    layers = [l for l in confs['layers'] if l['name'] in kept_layers]
    layers[1]['inbound_nodes'][0][0][0] = layers[0]['name']
    
    # set conf
    confs['layers'] = layers
    confs['input_layers'][0][0] = layers[0]['name']
    confs['output_layers'][0][0] = layers[-1]['name']
    
    # create new model
    submodel = tf.keras.Model.from_config(confs)
    for l in submodel.layers:
        orig_l = model.get_layer(l.name)
        if orig_l is not None:
            l.set_weights(orig_l.get_weights())
    return submodel

# vgg_16, vgg_19,  
# mobilenet_v2, xception, densenet_121, densenet_169, densenet_201, 
# efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, 
# efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

res = 260
input_size = (res, res, 1)

# List of hyperparameter possible values
hyperparameter_dict = { "num_epochs":                      50,  # Total number of training epochs
                        "batchsize":                        8,  # Minibatch size
                        "input_height":         input_size[0],  # Model's input height
                        "input_width":          input_size[1],  # Model's input width
                        "input_channels":       input_size[2],  # Model's input channels
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
                        "architecture":  "efficientnet_B6",  # Chosen architecture
                      }       

model_dir  = os.path.join( ".", "output", "models", "joao_123" )
model_path = os.path.join( model_dir, f"{hyperparameter_dict['architecture']}.h5" )

model_builder = ModelBuilder( model_path = model_path, gen_fig = True )
model_mine = model_builder( hyperparameter_dict, seed = 69 )

if not "v2" in hyperparameter_dict['architecture'].lower():
  model_keras = tf.keras.applications.EfficientNetB6( input_shape = input_size, include_top = True, classes = 1, weights = None,
                                                      classifier_activation='sigmoid' )
  model_keras = remove_normalization_layers(model_keras, start = 3)
    

else:
  model_keras = tf.keras.applications.EfficientNetV2B2( input_shape = input_size, include_top = True, classes = 1, weights = None,
                                                      classifier_activation='sigmoid', include_preprocessing = False )

path = os.path.join (".", "output", "models", "joao_123", f"efficientnet_keras.png" )

tf.keras.utils.plot_model( model_keras, to_file = path, show_shapes = True, show_layer_names = True, 
                            rankdir = "TB", expand_nested = False, dpi = 96 )
        
# Counts the model's parameters
trainable_count, non_trainable_count = ModelBuilder.count_model_params(model_keras)
print("\nCreated model with {:,} trainable parameters and {:,} non trainable ones...".format(trainable_count, non_trainable_count))

k_layer_idxs = [ i for i, layer in enumerate(model_keras.layers) if isinstance(layer, tf.keras.layers.MaxPooling2D) or is_conv(layer) or isinstance(layer, tf.keras.layers.Dense) ]
m_layer_idxs = [ i for i, layer in enumerate(model_mine.layers) if isinstance(layer, tf.keras.layers.MaxPooling2D) or is_conv(layer) or isinstance(layer, tf.keras.layers.Dense) ]
assert len(k_layer_idxs) == len(m_layer_idxs), f"{len(k_layer_idxs)} layers for Keras and {len(m_layer_idxs)} layers for mine"

input_txt = input()
if input_txt != "":

  # for l, (idx_k, idx_m) in enumerate(zip(k_layer_idxs, m_layer_idxs)):
    
  #   k_layer = model_keras.layers[idx_k]
  #   k_layer_type = k_layer.__class__.__name__
    
  #   m_layer = model_mine.layers[idx_m]
  #   m_layer_type = m_layer.__class__.__name__
    
  #   if m_layer_type != k_layer_type:
  #     print(f"{str(l).zfill(3)}: Keras: '{k_layer.name}' ({k_layer_type}), Mine: '{m_layer.name}' ({m_layer_type})")

  n_layers = 0
  for idx_k, idx_m in zip(k_layer_idxs, m_layer_idxs):
    layer_m = model_mine.layers[idx_m]
    layer_k = model_keras.layers[idx_k]
    
    n_layers = show_attr( layer_k, layer_m, n_layers, show = "diff" )

  # k_other_layers = [ layer for i, layer in enumerate(model_keras.layers) if (not i in k_layer_idxs) and has_params(layer) ]
  # trainable_count = int(np.sum([ count_params(l.trainable_weights) for l in k_other_layers ]))
  # non_trainable_count = int(np.sum([ count_params(l.non_trainable_weights) for l in k_other_layers ]))
  # print("\nKeras Other Layers have {:,} trainable parameters and {:,} non trainable ones...".format(trainable_count, non_trainable_count))

  # m_other_layers = [ layer for i, layer in enumerate(model_mine.layers) if (not i in m_layer_idxs) and has_params(layer) ]
  # trainable_count = int(np.sum([ count_params(l.trainable_weights) for l in m_other_layers ]))
  # non_trainable_count = int(np.sum([ count_params(l.non_trainable_weights) for l in m_other_layers ]))
  # print("\nMine Other Layers have {:,} trainable parameters and {:,} non trainable ones...".format(trainable_count, non_trainable_count))

  # k_trainable_count = 0
  # m_trainable_count = 0
  # k_non_trainable_count = 0
  # m_non_trainable_count = 0
  # for n_layers, (layer_k, layer_m) in enumerate(zip(k_other_layers, m_other_layers)):
  #   k_trainable = int(count_params(layer_k.trainable_weights))
  #   m_trainable = int(count_params(layer_m.trainable_weights))
  #   k_non_trainable = int(count_params(layer_k.non_trainable_weights))
  #   m_non_trainable = int(count_params(layer_m.non_trainable_weights))
    
  #   k_trainable_count += k_trainable
  #   m_trainable_count += m_trainable
  #   k_non_trainable_count += k_non_trainable
  #   m_non_trainable_count += m_non_trainable
    
  #   print(f"{str(n_layers+1).zfill(3)}: Keras: '{layer_k.name}', Mine: '{layer_m.name}'")
  #   print(f"\t{k_trainable_count}/{m_trainable_count} - Trainable - Keras: '{k_trainable}', Mine: '{m_trainable}'")
  #   print(f"\t{k_non_trainable_count}/{m_non_trainable_count} - Non Trainable - Keras: '{k_non_trainable}', Mine: '{m_non_trainable}'")
  #   print("\n")