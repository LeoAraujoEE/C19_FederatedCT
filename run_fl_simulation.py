import os
import sys

from utils.custom_models import ModelBuilder
from utils.federated_utils import FederatedServer
from utils.federated_utils import FederatedClient

# Decodes all the input args and creates a dict
arg_dict = FederatedServer.decode_args(sys.argv)

# 
PATH_DICT = { "datasets": arg_dict["data_path"],
              "outputs" : arg_dict["output_dir"],
            }

# List of available datasets
dataset_list = [ "Comp_CNCB_iCTCF", # 88k / 69k - Combination of CNCB non COVID samples + iCTCF
                 "miniCNCB",        # 74k / 55k - Rest of CNCB dataset
                 "COVID-CT-MD",     # 23k / 20k - COVID-CT-MD dataset
                 "Comp_LIDC-SB",    # 18k / 18k - Combination of LIDC + Stone Brook
                 "COVID-CTSet",     # 12k / 12k - COVID-CTSet dataset
               ]
  
federatedServer = FederatedServer( PATH_DICT, arg_dict )

# Initializes clients
for i, dataset in enumerate(dataset_list):
    
    # A new client is initialized for each dataset
    client_id = i+1
    client = FederatedClient(client_id, 
                             federatedServer.get_client_path_dict(), dataset, 
                             hyperparameters = federatedServer.hyperparameters,
                             aug_params = federatedServer.aug_params, 
                             keep_pneumonia = federatedServer.keep_pneumonia)
    
    # Gets the n° of train samples (used in model aggregation)
    client_train_sample_count = client.dataset.get_num_samples("train")
    
    # Creates a dict to store clients and another to store their sample count
    federatedServer.client_dict[client_id] = client
    federatedServer.num_samples_dict[client_id] = client_train_sample_count

# Creates and compiles the Model
global_model_path = federatedServer.create_global_model()

# Computes the number of aggregations
num_aggregations = federatedServer.get_num_aggregations()

for step in range(num_aggregations):
    # Selects clients to the current round
    print("\nStarting update round, selecting clients:")
    selected_ids = federatedServer.select_clients()
    
    # Computes the weight of each client's gradients for the aggregation
    client_weights = federatedServer.get_client_weights(selected_ids)
    
    # Computes the maximum amount of training steps allowed
    max_train_steps = federatedServer.get_max_train_steps(selected_ids)
    
    # Gets the index of the current epoch and the number of epochs to
    # be performed by each local model
    current_epoch, step_num_epochs = federatedServer.get_epoch_info(step)
    
    # Dict to register model paths and number of samples
    local_model_paths = {}
    for client_id in selected_ids:
        
        # Trains a local model for the current selected client
        client = federatedServer.client_dict[client_id]
        local_model_path = client.run_train_process(global_model_path, 
                                      step, epoch_idx = current_epoch,
                                      num_epochs = step_num_epochs, 
                                      max_train_steps = 10,
                                      # max_train_steps = max_train_steps,
                                      ignore_check = arg_dict["ignore_check"])
        
        # Appends the path and n° of samples to the dict
        local_model_paths[client_id] = local_model_path
    
    # Updates global model
    global_model_path = federatedServer.update_global_model(local_model_paths,
                                                         client_weights, step)
    
    # Evaluates updated global model on validation data
    federatedServer.run_eval_process( step, test = False )
    
    # Updates current epoch number
    current_epoch += federatedServer.fl_params["epochs_per_step"]
