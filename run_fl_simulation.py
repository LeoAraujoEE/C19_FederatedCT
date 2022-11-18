import sys
import json
import time
from utils.federated_utils import FederatedServer
from utils.federated_utils import FederatedClient

# Decodes all the input args and creates a dict
arg_dict = json.loads(sys.argv[1])

# 
PATH_DICT = { "datasets": arg_dict.pop("data_path"),
              "outputs" : arg_dict.pop("output_dir"),
            }

# List of available datasets
DATASETS = [ "Comp_CNCB_iCTCF", # 88k / 69k - Combination of CNCB non COVID samples + iCTCF
                    "miniCNCB", # 74k / 55k - Rest of CNCB dataset
                 "COVID-CT-MD", # 23k / 20k - COVID-CT-MD dataset
                "Comp_LIDC-SB", # 18k / 18k - Combination of LIDC + Stone Brook
                 "COVID-CTset", # 12k / 12k - COVID-CTSet dataset
           ]

# Extract info from from args_dict
val_dset        = arg_dict.pop("dataset")
model_id        = arg_dict.pop("model_hash")
ignore_check    = arg_dict.pop("ignore_check")
keep_pneumonia  = arg_dict.pop("keep_pneumonia")
model_fname     = arg_dict.pop("model_filename")
fl_params       = arg_dict.pop("fl_params")
hyperparameters = arg_dict.pop("hyperparameters")
data_aug_params = arg_dict.pop("data_augmentation")

# Initializes federatedServer
federatedServer = FederatedServer(PATH_DICT, model_fname, model_id, val_dset,
                                  fl_params, hyperparameters, data_aug_params,
                                  keep_pneumonia, ignore_check)

if federatedServer.check_step( ignore_check ):

    # Removes models whose training process did not finish properly
    federatedServer.prepare_model_dir()

    # Measures time at the start of the Federated Training process
    fl_init_time = time.time()

    # Initializes clients
    print("\nInitializing clients:")
    for i, dataset in enumerate(DATASETS):
        
        # A new client is initialized for each dataset
        client_id = i+1
        client = FederatedClient(client_id, 
                                federatedServer.get_client_path_dict(), dataset, 
                                hyperparameters = federatedServer.hyperparameters,
                                aug_params = federatedServer.aug_params, 
                                keep_pneumonia = federatedServer.keep_pneumonia,
                                ignore_check = federatedServer.ignore_check)
        
        # Gets the n° of train samples (used for model aggregation)
        client_train_sample_count = client.dataset.get_num_samples("train")
        
        # Creates a dict to store clients and another to store their sample count
        federatedServer.client_dict[client_id] = client
        federatedServer.num_samples_dict[client_id] = client_train_sample_count

    # Number of steps in the Federated Learning process
    total_steps = federatedServer.get_num_steps()

    # Federated Learning loop
    for step in range(total_steps):
        # Step starting time
        step_t0 = time.time()
        
        # If its the first step, creates and compiles the Model
        if step == 0:
            global_model_path = federatedServer.create_global_model()
        
        # Otherwise, trains local models to update the global model
        else:
            # 
            client_weights, local_paths = federatedServer.train_local_models(step,
                                                                    total_steps)
            
            # Updates global model
            global_model_path = federatedServer.update_global_model(local_paths,
                                                        client_weights, step)
        
        # Passes the global model to all clients & gets their train/val metrics
        monitored_val = federatedServer.validate_global_model(global_model_path, 
                                                            step, total_steps)
        
        # Checks wether the new global model has the best results so far
        # If so, updates the main weights file with the current weights
        federatedServer.model_checkpoint.create_checkpoint(monitored_val, 
                                                    global_model_path)
        
        # Prints the total time to complete the current step
        print(f"\nCompleted Federated Learning Step #{step}/{total_steps} in",
              f" {int(time.time()-step_t0)} seconds...")
        
        # Stops the training if EarlyStopping conditions are met
        if federatedServer.early_stopping.is_triggered(monitored_val, step):
            break

    # Deletes all temporary models to keep only the final checkpoint
    federatedServer.clear_tmp_models()
            
    # Measures total training time
    fl_ellapsed_time = (time.time() - fl_init_time)
    fl_train_time = federatedServer.ellapsed_time_as_str(fl_ellapsed_time)

    # 
    federatedServer.plot_train_results()

    #
    print("\nSaving training hyperparameters as JSON...")
    federatedServer.hyperparam_to_json(hyperparameters, data_aug_params, 
                                    fl_train_time, fl_params)
        
    # Tests the final selected global model
    federatedServer.run_test_process()
