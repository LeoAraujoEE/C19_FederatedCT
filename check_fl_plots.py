import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params

NUM_SAMPLES = 10
base_path = "output\federated_models\dropped\fl_resnet18_43087dfbd0b9787529f43892551e8d9a__2"

def fetch_random_params(weights, num_samples, seed):
    n_values = np.prod(weights.shape)
    w_reshaped = list(weights.reshape( (n_values,) ))
    
    if num_samples >= n_values:
        return w_reshaped
    
    np.random.seed(seed)
    return np.random.choice(w_reshaped, num_samples, replace = False)

def count_weights(layer):
    num_weights = 0
    if hasattr(layer, "trainable_weights"):
        num_weights += int(count_params(layer.trainable_weights))
    if hasattr(layer, "non_trainable_weights"):
        num_weights += int(count_params(layer.non_trainable_weights))
    return num_weights

def load_model( model_path ):
    config_path = model_path.replace(".h5", ".json")
    # Opening JSON file
    with open( config_path ) as json_file:
        json_config = json.load(json_file)

    # Loads model from JSON configs and H5 or Tf weights
    model = tf.keras.models.model_from_json(json_config)
    model.load_weights( model_path )
    return model
    
def federated_average(local_model_paths, client_weights):
    
    # Gets weights from each trained model
    model_weights = []
    for local_path in local_model_paths.values():
        local_model = load_model(local_path)
        model_weights.append( local_model.get_weights() )
    
    # Gets weights for each client based on their available samples
    client_weights = [w for w in client_weights.values()]
    
    # List of new global model weights
    new_global_weights = []
    
    # Iterates 'model_weights' returning 
    # tuples of weight arrays from each selected client 
    for weights_array_tuple in zip(*model_weights):
        
        # List for new weights of the updated global model
        new_weights_list = []
    
        # Iterates 'weights_array_tuple' returning 
        # tuples of weights from each selected client
        for weights in zip(*weights_array_tuple):
            
            new_weights_list.append( np.average(np.array(weights), axis = 0, 
                                                weights = client_weights) )
        
        new_global_weights.append( np.array(new_weights_list) )
    
    return new_global_weights

# Client Weights (#samples in train partition)
client_weights = { 1: 41857, 2: 32942, 3: 12554, 4: 11071, 5: 7234 }

# Path to resulting weights before 1st aggregation
model_paths, local_models = {}, {}
for i in range(1,6):
    model_paths[i] = os.path.join( base_path, "local", f"client_0{i}", f"local_model_{i}_v0", f"local_model_{i}_v0.h5")
    # local_models[i] = load_model(os.path.join( base_path, "global", "global_model_v0", "global_model_v0.h5"))
    local_models[i] = load_model(model_paths[i])

# Global model with initial random weights
global_model = load_model(os.path.join( base_path, "global", "global_model_v1", "global_model_v1.h5"))

# Number of layers
num_layers = len(global_model.layers)

# for idx in range(num_layers):
for idx in range(50):
    str_idx = str(idx+1).zfill(3)
    layer_name = global_model.layers[idx].name
    num_weights = count_weights(global_model.layers[idx])
    
    if num_weights == 0:
        continue
    
    print(f"[{str_idx}] Layer '{layer_name}': {num_weights})")
    
    weights = global_model.layers[idx].get_weights()
    n_idxs  = np.min([len(weights), 3])
    w_idxs  = np.random.choice(len(weights), n_idxs, replace = False)
    
    for w_idx in w_idxs:
        w = weights[w_idx]
        sd = np.random.randint(255)
        print( f"\tSet of weights #{w_idx+1} - Shape {w.shape}..." )
        
        # Sampled weights for global model
        sampled_weights = fetch_random_params(w, NUM_SAMPLES, sd)
        print("\t\t[GLOBAL] - ", " ".join([f"{n:.3E}" if n < 0 else f"+{n:.3E}" for n in sampled_weights]))
        
        fed_weights = [0 for _ in sampled_weights]
        for local_id, local_model in local_models.items():
            local_layer = local_model.get_layer(layer_name)
            local_weights = local_layer.get_weights()
        
            # Sampled weights for global model
            local_w = local_weights[w_idx]
            local_sampled_weights = fetch_random_params(local_w, NUM_SAMPLES, sd)
            print(f"\t\t[LOCAL{local_id}] - ", " ".join([f"{n:.3E}" if n < 0 else f"+{n:.3E}" for n in local_sampled_weights]))
            
            for f_idx, n in enumerate(local_sampled_weights):
                n_att = client_weights[local_id] * n
                fed_weights[f_idx] += n_att
                
        total_weights = np.sum(list(client_weights.values()))
        fed_weights = [n/total_weights for n in fed_weights]
        print("\t\t[GLOBAL] - ", " ".join([f"{n:.3E}" if n < 0 else f"+{n:.3E}" for n in fed_weights]))
        print("\n")