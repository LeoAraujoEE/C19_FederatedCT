import os
import glob
import json
import time
import shutil
import random
import logging
import warnings
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf

# 
from utils.dataset import Dataset
from utils.custom_plots import CustomPlots
from utils.custom_models import ModelBuilder
from utils.custom_model_trainer import ModelHandler
from utils.custom_model_trainer import ModelManager

class FederatedServer(ModelHandler):
    def __init__(self, path_dict, model_fname, model_id, val_dataset, 
                 hyperparams, aug_params, keep_pneumonia, ignore_check):
        
        # Inherits ModelHandler's init method
        ModelHandler.__init__( self, path_dict["outputs"], model_fname, 
                               model_id )
        
        # Prints the selected hyperparameters
        print("\nUsing the current hyperparameters for Federated Learning:")
        self.hyperparameters = hyperparams
        self.print_dict(self.hyperparameters)
        
        # Prints the given data augmentation parameters
        print("\nUsing the current parameters for Data Augmentation:")
        self.aug_params = aug_params
        self.print_dict(self.aug_params)
        
        # Random number generator create from np.random
        self.rng = np.random.default_rng(seed = self.hyperparameters["seed"])
        
        # Directory for all available datasets
        self.data_path = path_dict["datasets"]
        
        # Wether to keep pneumonia sample or remove them
        self.keep_pneumonia = keep_pneumonia
        
        #
        self.ignore_check = ignore_check
        
        # Dataset used to validate models
        self.dataset = Dataset( self.data_path, name = val_dataset, 
                                keep_pneumonia = self.keep_pneumonia )
        
        # Initializes an empty client dict
        self.client_dict = {}
        
        # Initializes an empty dict to store client's training sample count
        self.num_samples_dict = {}

        return
    
    def select_clients(self):
        # Checks if there're at least 2 clients to simulate Federated Learning
        total_clients = len(self.client_dict)
        assert total_clients > 1, "Insuficient number of clients..."
        
        # Computes the number of clients to be selected
        client_frac = self.hyperparameters["client_frac"]
        num_selected = np.round( client_frac * total_clients, decimals = 0 )
        
        # The selected number is increased to 2 if it's smaller than 2
        num_selected = int(np.max([num_selected, 2]))
        
        # Selects client_ids randomly without replacement
        client_ids = list(self.client_dict.keys())
        selected_ids = self.rng.choice( client_ids, size = num_selected, 
                                        replace = False, shuffle = False )
        return selected_ids
    
    def create_global_model(self):
        
        # Filename for the current version of the global model
        current_fname = "global_model_v0"
        
        # Folder to contain current version of global model
        current_dst_dir = os.path.join( self.model_dir, "global", 
                                        current_fname )
        
        # Creates directory if needed
        if not os.path.exists(current_dst_dir):
            os.makedirs(current_dst_dir)
        
        # Path to the initial global model
        current_path = os.path.join(current_dst_dir, f"{current_fname}.h5")
        print(f"\nSaving global model to '{current_path}'...")

        # Creates the model
        model_builder = ModelBuilder( model_path = current_path )
        model = model_builder( self.hyperparameters, 
                               seed = self.hyperparameters["seed"] )
        
        # Saves model config & model weights
        self.save_model(model, current_path)
        
        return current_path
    
    def get_epoch_info( self, step_idx ):
        
        # Retrieves total n° of epochs / n° of epochs between aggregations
        epochs_per_step = self.hyperparameters["epochs_per_step"]
        total_epoch_num = self.hyperparameters["num_epochs"]
        
        # Gets the number of epochs were executed so far
        epoch_idx = step_idx * epochs_per_step
        
        # Gets the number of epochs to perform in the current step
        step_epochs = epochs_per_step
        if (epoch_idx + step_epochs) > total_epoch_num:
            # This value is corrected if it would surpass the 
            # total amount of epochs from self.hyperparameters
            step_epochs = (total_epoch_num - epoch_idx)
        
        return epoch_idx, step_epochs
    
    @staticmethod
    def save_model(model, model_path):

        # Saves model configs
        json_config = model.to_json()
        config_path = model_path.replace(".h5", ".json")

        with open(config_path, "w") as json_file:
            json.dump( json_config, json_file, indent=4 )
        
        # Saves model weights
        model.save_weights( model_path )
        
        return
    
    def federated_average(self, local_model_paths, client_weights):
        
        # Gets weights from each trained model
        model_weights = []
        for local_path in local_model_paths.values():
            local_model = self.load_model(local_path)
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
    
    def update_global_model( self, local_model_paths, client_weights, step ):
        print("\nAggregating models:")
        
        # 
        old_fname = f"global_model_v{step}"
        old_path  = os.path.join(self.model_dir, "global", old_fname, 
                                 f"{old_fname}.h5")
        
        if self.hyperparameters['aggregation'].lower() == "fed_avg":
            global_model_weights = self.federated_average( local_model_paths,
                                                           client_weights )
        
        else:
            client_weights = { k: 1 for k in client_weights.keys() }
            global_model_weights = self.federated_average( local_model_paths,
                                                           client_weights )
        
        # Loads old model and replaces weights
        model = self.load_model(old_path)
        model.set_weights(global_model_weights)
        
        # Folder to contain current version of global model
        new_fname = f"global_model_v{step+1}"
        new_dst_dir = os.path.join(self.model_dir, "global", new_fname)
        if not os.path.exists(new_dst_dir):
            os.makedirs(new_dst_dir)
        
        # Path to the initial global model
        new_model_path = os.path.join(new_dst_dir, f"{new_fname}.h5")
        print(f"\nSaving global model to '{new_model_path}'...")
        
        # Saves model config & model weights
        self.save_model(model, new_model_path)
        
        return new_model_path
    
    def get_final_model(self):
        # Path to CSV file w/ val metrics for all versions of the global model
        tmp_csv_path = os.path.join(self.model_dir, "global", 
                                    "training_results.csv")

        # If the CSV file already exists
        if os.path.exists(tmp_csv_path):
            # Loads the old file
            tmp_df = pd.read_csv(tmp_csv_path, sep = ";")
        
        # Gets the best performing version of global model 
        # based on the values for the monitored variable
        var_list = ["val_loss", "val_acc", "val_f1"]
        if self.hyperparameters["monitor"] in var_list:
            # Gets the column name for the monitored variable in tmp_df
            # (this name differs based on the validation dataset used)
            dataset_name = self.dataset.name.lower()
            monitor_var  = self.hyperparameters["monitor"]
            column_name  = monitor_var.replace("val", dataset_name)
            
            # Locates the row w/ best value from dataframe
            if "loss" in monitor_var:
                # Which is the min value for Loss
                row_idx = tmp_df[column_name].idxmin()
                row_bst = tmp_df[column_name].min()
            else:
                # And max value for Acc/F1
                row_idx = tmp_df[column_name].idxmax()
                row_bst = tmp_df[column_name].max()
                
            # Extracts the corresponding row
            row = tmp_df.iloc[row_idx]
            sufix = f"has the best '{column_name}' of {row_bst:.4f}..."
            
        # If not monitoring a variable, gets the most recent version
        else:
            # Extracts the last row from dataframe
            row = tmp_df.iloc[-1]
            sufix = "is the most recent..."
            
        # Gets model path
        src_model_weights_path = row["model_path"]
        
        # Gets the src/dst path to the selected model's config
        dst_model_cofigs_path = self.model_path.replace(".h5",".json")
        src_model_configs_path = src_model_weights_path.replace(".h5",".json")
        
        # Copies the selected model's weights and configs
        print(f"\nSelected model '{src_model_weights_path}', which {sufix}")
        shutil.copy2(src_model_weights_path, self.model_path)
        shutil.copy2(src_model_configs_path, dst_model_cofigs_path)
        
        return self.model_path
    
    def get_client_weights(self, selected_ids):
    
        samples = { _id: self.num_samples_dict[_id] for _id in selected_ids }
        total_samples = np.sum(list(samples.values()))
        
        weights = { _id: s / total_samples  for _id, s in samples.items() }
        
        return weights
    
    def get_max_train_steps(self, selected_ids):
        # Extracts relevant hyperparameters from hyperparameters dict
        batchsize = self.hyperparameters["batchsize"]
        undersampling = self.hyperparameters["apply_undersampling"]
        
        # Lists the maximum number of training steps (batches) 
        # that each of the selected client can produce
        num_step_list = []
        for _id in selected_ids:
            client = self.client_dict[_id]
            num_step = client.dataset.get_num_steps( "train", 
                                                     batchsize, 
                                                     undersampling )
            
            num_step_list.append(num_step)
        
        # Extracts the minimum and the maximum values from that list
        min_steps = np.min(num_step_list)
        max_steps = np.max(num_step_list)
        
        # Computes the 'max_train_steps' as a value between min_steps and
        # max_steps that's closer to min_steps as 'max_steps_frac' is closer
        # to 0 and is closer to max_steps as 'max_steps_frac' is closer to 1
        steps_frac = self.hyperparameters["max_steps_frac"]
        xtra_steps = (max_steps - min_steps) * steps_frac
        max_train_steps = int(min_steps + xtra_steps)
        
        return max_train_steps
    
    def get_num_aggregations(self):
        # Retrieves n° of epochs / n° of epochs per aggregation
        epochs_per_step = self.hyperparameters["epochs_per_step"]
        total_epoch_num = self.hyperparameters["num_epochs"]
        
        # Returns the number of aggregations
        return int(np.ceil( total_epoch_num / epochs_per_step ))
    
    def get_client_path_dict(self):
        path_dict = { "datasets"      : self.data_path,
                      "local_outputs" : os.path.join(self.model_dir, "local"),
                      "global_outputs": os.path.join(self.model_dir, "global")
                    }
        return path_dict
    
    def get_val_args(self, step):
        
        model_id = f"v{step+1}"
        model_fname = f"global_model_v{step+1}"
        global_model_dir = os.path.join(self.model_dir, "global")
        
        # Base args dict
        args = { "output_dir"       :             global_model_dir, 
                 "data_path"        :               self.data_path,
                 "dataset"          :                         None, 
                 "keep_pneumonia"   :          self.keep_pneumonia,
                 "ignore_check"     :            self.ignore_check,
                 "model_hash"       :                     model_id, 
                 "model_filename"   :                  model_fname,
                 "eval_partition"   :                        "val",
                 "hyperparameters"  :         self.hyperparameters,
                 "data_augmentation":              self.aug_params,
                 "verbose"          :                            0,
                 "seed"             : self.hyperparameters["seed"],
               }
        
        return args
    
    def get_test_args(self):
        
        # Base args dict
        args = { "output_dir"       :                 self.dst_dir, 
                 "data_path"        :               self.data_path,
                 "dataset"          :       self.dataset.orig_name, 
                 "keep_pneumonia"   :          self.keep_pneumonia,
                 "ignore_check"     :            self.ignore_check,
                 "model_hash"       :                self.model_id, 
                 "model_filename"   :             self.model_fname,
                 "eval_partition"   :                       "test",
                 "hyperparameters"  :         self.hyperparameters,
                 "data_augmentation":              self.aug_params,
                 "verbose"          :                            0,
                 "seed"             : self.hyperparameters["seed"],
               }
        
        return args
    
    def run_eval_process( self, step, test ):
        # Different args are selected to validate the current global model
        # or to test the final selected global model
        args = self.get_test_args() if test else self.get_val_args(step)     
        
        # Serializes args dict as JSON formatted string
        serialized_args = json.dumps(args)
        
        # 
        command = ["python", "-m", "run_model_testing", serialized_args]

        # Evaluates model
        subprocess.Popen.wait(subprocess.Popen( command ))
        return

class FederatedClient(ModelManager):
    def __init__(self, id, path_dict, dataset_name, hyperparameters = None, 
                 aug_params = None, keep_pneumonia = False):
        
        #
        self.client_id = id
        self.client_name = f"client_{str(self.client_id).zfill(2)}"
        
        # Directory for all available datasets
        self.data_path = path_dict["datasets"]
        
        # Directory for the output trained models
        self.dst_dir = os.path.join(path_dict["local_outputs"], 
                                    self.client_name)
        
        # 
        self.global_dst_dir = path_dict["global_outputs"]
        
        # Hyperparameters used for training
        self.hyperparameters = hyperparameters
        
        # Data augmentation parameters used for training
        self.aug_params = aug_params
        
        # Wether to keep pneumonia sample or remove them
        self.keep_pneumonia = keep_pneumonia
        
        # Builds object to handle the training dataset
        self.dataset = Dataset( self.data_path, name = dataset_name, 
                                keep_pneumonia = self.keep_pneumonia )
            
        
        # Reports client's creation
        print(f"\tCreated client #{self.client_id} w/ dataset '{self.dataset.name}'...")

        return

    def run_train_process( self, global_model_path, step_idx, num_epochs,
                           epoch_idx, max_train_steps, ignore_check ):
        
        # Combines client_id and communication round to make local model_id
        # The model filename combines the architecture used and the model_id
        model_id = f"{self.client_id}_v{step_idx}"
        model_fname = f"local_model_{model_id}"
        
        # Base args dict
        args = { "output_dir"       :                 self.dst_dir, 
                 "data_path"        :               self.data_path,
                 "dataset"          :       self.dataset.orig_name, 
                 "keep_pneumonia"   :          self.keep_pneumonia,
                 "ignore_check"     :                 ignore_check,
                 "model_hash"       :                     model_id, 
                 "model_filename"   :                  model_fname,
                 "initial_weights"  :            global_model_path,
                 "max_train_steps"  :              max_train_steps,
                 "epochs_per_step"  :                   num_epochs,
                 "current_epoch_num":                    epoch_idx,
                 "hyperparameters"  :         self.hyperparameters,
                 "data_augmentation":              self.aug_params,
                 "verbose"          :                            0,
                 "seed"             : self.hyperparameters["seed"],
               }
        
        # Serializes args dict as JSON formatted string
        serialized_args = json.dumps(args)
        
        # 
        command = ["python", "-m", "run_model_training", serialized_args]

        # Trains model
        subprocess.Popen.wait(subprocess.Popen( command ))

        # Gets the path to the trained model
        local_model_path = os.path.join( self.dst_dir, model_fname, 
                                         f"{model_fname}.h5" )
        
        # Returns the path to the trained model and n° of train samples
        return local_model_path
    
    def update_history_dict(self, global_model_path, step_idx):
            
        # Gets the filename for the current step's local model
        model_id = f"{self.client_id}_v{step_idx}"
        model_fname = f"local_model_{model_id}"
        
        # Path to current local model's history dict, 
        # which only records this step's metrics
        src_history_path = os.path.join(self.dst_dir, model_fname, 
                                        "history_dict.csv")
        
        # Path to the client's history dict,
        # which records metrics for all steps this client has participated
        dst_history_path = os.path.join(self.dst_dir, "history_dict.csv")
        
        # If the client's history dict doesnt exist yet,
        # if not os.path.exists(dst_history_path):
        if step_idx == 0:
            # The current step's history dict is simply copied
            updated_df = pd.read_csv( src_history_path, sep = ";" )
            step_list = [f"{step_idx+1}.{i+1}" for i in range(len(updated_df))]
  
            # Adds a new column for the current Step/Epoch
            updated_df.insert(0, "Step.Epoch", step_list)
        
        else:
            # Loads and concatenates both dataframes
            src_df = pd.read_csv( src_history_path, sep = ";" )
            dst_df = pd.read_csv( dst_history_path, sep = ";" )
            step_list = [f"{step_idx+1}.{i+1}" for i in range(len(src_df))]
  
            # Adds a new column for the current Step/Epoch
            src_df.insert(0, "Step.Epoch", step_list)

            # Appends the new dataframe as extra rows
            updated_df = pd.concat( [dst_df, src_df], ignore_index = True )
        
        # Saves the dataframe as CSV
        updated_df.to_csv( dst_history_path, index = False, sep = ";" )
        
        return