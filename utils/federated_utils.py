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
from utils.custom_plots import CustomPlots
from utils.custom_models import ModelBuilder
from utils.custom_model_trainer import ModelEntity
from utils.custom_model_trainer import ModelTrainer
from utils.custom_model_trainer import ModelManager

class FederatedServer(ModelEntity):
    def __init__(self, path_dict, args_dict):
        
        # Random number generator create from np.random
        self.rng = np.random.default_rng(seed = args_dict["seed"])
        
        # Directory for all available datasets
        self.data_path = path_dict["datasets"]
        
        # Output directory
        self.dst_dir = path_dict["outputs"]
  
        # Removes models whose training process did not finish properly
        self.remove_unfinished()
        
        # Current models' directory inside dst_dir
        # Generates the model name and the path to its dir
        self.model_path = self.gen_model_path(args_dict)
        self.model_dir, self.model_fname = os.path.split(self.model_path)
        
        # Wether to keep pneumonia sample or remove them
        self.keep_pneumonia = args_dict["keep_pneumonia"]
        
        # Extracts FL parameters/Hyperparameters/DataAugmentation parameters
        fl_params, hyperparams, aug_params = self.get_dicts(args_dict)
        
        # Prints the possible values
        print("\nUsing the current parameters for Federated Learning:")
        self.fl_params = fl_params
        self.print_dict(self.fl_params)
        
        # Prints the possible values
        print("\nUsing the current Hyperparameters:")
        self.hyperparameters = hyperparams
        self.print_dict(self.hyperparameters)
        
        # Prints the given data augmentation parameters
        print("\nUsing the current parameters for Data Augmentation:")
        self.aug_params = aug_params
        self.print_dict(self.aug_params)
        
        # Initializes an empty client dict
        self.client_dict = {}

        return

    def remove_unfinished(self):

        # Path to CSV file
        csv_path = os.path.join( self.dst_dir, "fl_training_results.csv" )

        # Returns True if the csv file does not exist yet
        finished_models = []
        if os.path.exists( csv_path ):
            results_df = pd.read_csv(csv_path, sep = ";")
            finished_models = results_df["model_path"].to_list()

        # Lists all model subdirs in self.dst_dir
        all_subdirs = glob.glob(os.path.join(self.dst_dir, "*"))
        all_subdirs = sorted([p for p in all_subdirs if os.path.isdir(p)])

        # Iterates through those files to check .h5 path in the CSV
        # Which indicates that train/test process finished correctly
        for path2subdir in all_subdirs:
            model_basename = os.path.split(path2subdir)[-1]
            weights_path = os.path.join(path2subdir, f"{model_basename}.h5")

            if weights_path in finished_models:
                continue

            print(f"Deleting '{model_basename}' subdir as its training did not finish properly...")
            shutil.rmtree(path2subdir, ignore_errors=False)
        return

    def gen_model_path(self, args_dict):

        # Extract model's hash and model's filename from args_dict
        model_id = args_dict["model_hash"]
        model_fname = args_dict["model_filename"]

        # Creates the full model path
        model_path = os.path.join( self.dst_dir, model_fname, 
                                   f"{model_fname}.h5" )
        
        # Checks if a model with the same name already exists
        # possible if a combination of hyperparameters is being retrained
        if os.path.exists( os.path.dirname(model_path) ):
            # Path to results CSV file
            csv_path = os.path.join(self.dst_dir, "fl_training_results.csv")
            
            idx = 0
            if os.path.exists(csv_path):
                # Reads CSV file to count models w/ same hash
                result_df = pd.read_csv(csv_path, sep = ";")

                # Counts the amount of models with the same hash
                idx = len(result_df[result_df["model_hash"] == model_id])
            
            # Keeps the same path / fname if there're no entries
            if idx > 0:
                # Otherwise, updates model_fname and model_path 
                # to avoid overwritting the existant model
                model_fname = f"{model_fname}_{idx+1}"
                model_path = os.path.join(self.model_dir, model_fname,
                                          f"{model_fname}.h5")

        return model_path
    
    def create_global_model(self):
        
        # Folder to contain all versions of global model
        dst_dir = os.path.join(self.model_dir, "global")
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        # Path to the initial global model
        model_fname = self.model_fname.replace(".h5", "_v0.h5") 
        model_path = os.path.join(dst_dir, model_fname)
        
        # 
        print(f"\nSaving global model to '{model_path}'...")

        # Creates the model
        model_builder = ModelBuilder( model_path = model_path )
        model = model_builder( self.hyperparameters, 
                               seed = self.hyperparameters["seed"] )
        
        # Test this model
        # Add model to 'csv'
        
        # Saves model weights
        model.save_weights( model_path )
        
        return model_path
    
    def select_clients(self):
        # Checks if there're at least 2 clients to simulate Federated Learning
        total_clients = len(self.client_dict)
        assert total_clients > 1, "Insuficient number of clients..."
        
        # Computes the number of clients to be selected
        client_frac = self.fl_params["client_frac"]
        num_selected = np.round( client_frac * total_clients, decimals = 0 )
        
        # The selected number is increased to 2 if it's smaller than 2
        num_selected = int(np.max([num_selected, 2]))
        
        # Selects client_ids randomly without replacement
        client_ids = list(self.client_dict.keys())
        selected_ids = self.rng.choice( client_ids, size = num_selected, 
                                        replace = False, shuffle = False )
        return selected_ids

    def get_dicts(self, args_dict):
        
        # Generates fedlearn_params through the default values,
        # them updates the values with the ones available in args_dict
        fedlearn_params = self.get_default_fl_params()
        fedlearn_params = self.update_dict_values(args_dict, fedlearn_params)
        
        # Generates hyperparameters through the default values,
        # them updates the values with the ones available in args_dict
        hyperparameters = self.get_default_hyperparams()
        hyperparameters = self.update_dict_values(args_dict, hyperparameters)
        
        # Generates data_aug_params through the default values,
        # them updates the values with the ones available in args_dict
        data_aug_params = self.get_default_augmentations()
        data_aug_params = self.update_dict_values(args_dict, data_aug_params)
        
        # Returns both dicts
        return fedlearn_params, hyperparameters, data_aug_params
    
    def get_num_aggregations(self):
        # Retrieves n° of epochs / n° of epochs per aggregation
        epochs_per_step = self.fl_params["epochs_per_step"]
        total_epoch_num = self.hyperparameters["num_epochs"]
        
        # Returns the number of aggregations
        return int(np.ceil( total_epoch_num / epochs_per_step ))
    
    def get_client_path_dict(self):
        path_dict = { "datasets": self.data_path,
                      "outputs" : os.path.join(self.model_dir, "local") }
        return path_dict

class FederatedClient(ModelEntity):
    def __init__(self, id, path_dict, dataset_name, hyperparameters = None, 
                 aug_params = None, keep_pneumonia = False):
        
        #
        self.client_id = id
        
        # Directory for all available datasets
        self.data_path = path_dict["datasets"]
        
        # Directory for the output trained models
        self.dst_dir = path_dict["outputs"]
        
        # Name of the selected train dataset
        self.dataset_name = dataset_name
        
        # Path to global model
        # self.global_model_path = model_path
        
        # Hyperparameters used for training
        self.hyperparameters = hyperparameters
        
        # Data augmentation parameters used for training
        self.aug_params = aug_params
        
        # Wether to keep pneumonia sample or remove them
        self.keep_pneumonia = keep_pneumonia
        
        # Reports client's creation
        print(f"\tCreated client #{self.client_id} w/ dataset '{self.dataset_name}'...")

        return

    def run_train_process(self, global_model_path, round_idx, ignore_check):
        
        # Combines client_id and communication round to make local model_id
        # The model filename combines the architecture used and the model_id
        model_id = f"{self.client_id}_v{round_idx}"
        model_fname = f"{self.hyperparameters['architecture']}_{model_id}"
        
        # Base args dict
        args = { "dataset"       :   self.dataset_name, 
                 "output_dir"    :        self.dst_dir, 
                 "data_path"     :      self.data_path,
                 "keep_pneumonia": self.keep_pneumonia,
                 "ignore_check"  :        ignore_check,
                 "model_hash"    :            model_id, 
                 "model_filename":         model_fname,
                 "load_from"     :   global_model_path,
                 "max_train_steps":                 10,
               }
        
        for dictionary in [self.aug_params, self.hyperparameters]:
            args.update(dictionary)
        
        command = ["python", "-m", "train_model"]
        for k, v in args.items():
            command.extend([self.get_flag_from_type(k, v), str(k), str(v)])

        # Trains model
        subprocess.Popen.wait(subprocess.Popen( command ))

        # Returns the path to the trained model
        local_model_path = os.path.join(self.dst_dir, model_fname)
        return local_model_path