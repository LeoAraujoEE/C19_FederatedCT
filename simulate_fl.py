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

# Creates and compiles the Model
global_model_path = federatedServer.create_global_model()

# Initializes clients
for i, dataset in enumerate(dataset_list):
    
    # A new client is initialized for each dataset
    client_id = i+1
    client = FederatedClient(client_id, 
                             federatedServer.get_client_path_dict(), dataset, 
                             hyperparameters = federatedServer.hyperparameters,
                             aug_params = federatedServer.aug_params, 
                             keep_pneumonia = federatedServer.keep_pneumonia)
    
    # 
    federatedServer.client_dict[client_id] = client

# Computes the number of aggregations
num_aggregations = federatedServer.get_num_aggregations()

current_epoch = 1
for step in range(num_aggregations):
    print("\nStarting update round, selecting clients:")
    
    # Lists the path to each local model produced
    local_model_list = []
    for client_id in federatedServer.select_clients():
        
        # Trains a local model for the current selected client
        client = federatedServer.client_dict[client_id]
        local_model_path = client.run_train_process(global_model_path, step,
                                    ignore_check = arg_dict["ignore_check"])
        
        # Appends the path to the trained local model
        local_model_list.append(local_model_path)
    
    assert all([os.path.exists(p) for p in local_model_list])
    print(f"\nStep #{step}/{num_aggregations}: All local models exist...")
    
    # Updates current epoch number
    current_epoch += federatedServer.fl_params["epochs_per_step"]
