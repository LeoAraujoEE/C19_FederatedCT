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
        self.model_path, self.model_id = self.gen_model_path(args_dict)
        self.model_fname = os.path.basename(self.model_path).replace(".h5","")
        self.model_dir = os.path.dirname(self.model_path)
        
        # Wether to keep pneumonia sample or remove them
        self.keep_pneumonia = args_dict["keep_pneumonia"]
        
        #
        self.ignore_check = args_dict["ignore_check"]
        
        # Dataset used to validate models
        self.dataset_name = args_dict["dataset"]
        self.dataset = Dataset( self.data_path, name = self.dataset_name, 
                                keep_pneumonia = self.keep_pneumonia )
        
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
        
        # Initializes an empty dict to store client's training sample count
        self.num_samples_dict = {}

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

        return model_path, model_id
    
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
    
    def create_global_model(self):
        
        #
        model_fname = "global_model_v0"
        
        # Folder to contain all versions of global model
        global_model_dir = os.path.join( self.model_dir, "global", 
                                         self.dataset.name )
        
        # Folder to contain all versions of global model
        dst_dir = os.path.join(global_model_dir, model_fname)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        # Path to the initial global model
        model_path = os.path.join(dst_dir, model_fname+".h5")
        print(f"\nSaving global model to '{model_path}'...")

        # Creates the model
        model_builder = ModelBuilder( model_path = model_path )
        model = model_builder( self.hyperparameters, 
                               seed = self.hyperparameters["seed"] )
        
        # Saves model config & model weights
        self.save_model(model, model_path)
        
        # Test this model
        # Add model to 'csv'
        
        return model_path
    
    def get_epoch_info( self, step_idx ):
        
        # Retrieves total n° of epochs / n° of epochs between aggregations
        epochs_per_step = self.fl_params["epochs_per_step"]
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
    
    @staticmethod
    def load_model(model_path):
        config_path = model_path.replace(".h5", ".json")
        
        # Opening JSON file
        with open( config_path ) as json_file:
            json_config = json.load(json_file)

        # Loads model from JSON configs and H5 or Tf weights
        model = tf.keras.models.model_from_json(json_config)
        model.load_weights( model_path )
        return model

    def get_model_weights_from_path( self, model_path ):
        print(f"\tLoading model '{model_path}'")
        # Loads model
        model = self.load_model(model_path)
        
        # Returns the model weights
        return model.get_weights()
    
    def federated_average(self, local_model_paths, client_weights):
        
        # Gets weights from each trained model
        model_weights = [self.get_model_weights_from_path(p) for p in local_model_paths.values()]
        
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
        print(f"\nAggregating models using {self.fl_params['aggregation']}:")
        
        # Folder to contain all versions of global model
        global_model_dir = os.path.join( self.model_dir, "global", 
                                         self.dataset.name )
        
        # 
        old_path = os.path.join(global_model_dir, f"global_model_v{step}", 
                                f"global_model_v{step}.h5")
        
        if self.fl_params['aggregation'].lower() == "fed_avg":
            global_model_weights = self.federated_average( local_model_paths,
                                                           client_weights )
        
        else:
            client_weights = { k: 1 for k in client_weights.keys() }
            global_model_weights = self.federated_average( local_model_paths,
                                                           client_weights )
        
        # Loads old model and replaces weights
        model = self.load_model(old_path)
        model.set_weights(global_model_weights)
        
        # Folder to contain all versions of global model
        dst_dir = os.path.join(global_model_dir, f"global_model_v{step+1}")
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        # Path to the initial global model
        new_model_path = os.path.join(dst_dir, f"global_model_v{step+1}.h5")
        print(f"\nSaving global model to '{new_model_path}'...")
        
        # Saves model config & model weights
        self.save_model(model, new_model_path)
        
        return new_model_path
    
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
            num_step = client.dataset.get_num_steps( "train", batchsize, 
                                                     undersampling )
            
            num_step_list.append(num_step)
        
        # Extracts the minimum and the maximum values from that list
        min_steps = np.min(num_step_list)
        max_steps = np.max(num_step_list)
        
        # Computes the 'max_train_steps' as a value between min_steps and
        # max_steps that's closer to min_steps as 'max_steps_frac' is closer
        # to 0 and is closer to max_steps as 'max_steps_frac' is closer to 1
        xtra_steps = (max_steps-min_steps) * self.fl_params["max_steps_frac"]
        max_train_steps = int(min_steps + xtra_steps)
        
        return max_train_steps

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
    
    def get_val_args(self, step):
        
        model_id = f"v{step+1}"
        model_fname = f"global_model_v{step+1}"
        global_model_dir = os.path.join(self.model_dir, "global")
        
        # Base args dict
        args = { "dataset"          :     self.dataset_name, 
                 "output_dir"       :      global_model_dir, 
                 "data_path"        :        self.data_path,
                 "keep_pneumonia"   :   self.keep_pneumonia,
                 "ignore_check"     :     self.ignore_check,
                 "model_hash"       :              model_id, 
                 "model_filename"   :           model_fname,
                 "load_from"        :                  None,
                 "max_train_steps"  :                  None,
                 "epochs_per_step"  :                  None,
                 "current_epoch_num":                  None,
                 "eval_partition"   :                 "val",
               }
        
        return args

    def run_eval_process( self, step, test = True ):
        
        if test:
            model_fname = self.model_fname.replace('.h5', '')
            
            # Base args dict
            args = { "dataset"          :     self.dataset_name, 
                     "output_dir"       :        self.model_dir, 
                     "data_path"        :        self.data_path,
                     "keep_pneumonia"   :   self.keep_pneumonia,
                     "ignore_check"     :     self.ignore_check,
                     "model_hash"       :         self.model_id, 
                     "model_filename"   :           model_fname,
                     "load_from"        :                  None,
                     "max_train_steps"  :                  None,
                     "epochs_per_step"  :                  None,
                     "current_epoch_num":                  None,
                     "eval_partition"   :                "test",
                   }
        
        else:
            args = self.get_val_args(step)
        
        for dictionary in [self.aug_params, self.hyperparameters]:
            args.update(dictionary)
        
        command = ["python", "-m", "test_model"]
        for k, v in args.items():
            command.extend([self.get_flag_from_type(k, v), str(k), str(v)])

        # Evaluates model
        subprocess.Popen.wait(subprocess.Popen( command ))
        return

class FederatedClient(ModelEntity):
    def __init__(self, id, path_dict, dataset_name, hyperparameters = None, 
                 aug_params = None, keep_pneumonia = False):
        
        #
        self.client_id = id
        
        # Directory for all available datasets
        self.data_path = path_dict["datasets"]
        
        # Directory for the output trained models
        self.dst_dir = path_dict["outputs"]
        
        # Hyperparameters used for training
        self.hyperparameters = hyperparameters
        
        # Data augmentation parameters used for training
        self.aug_params = aug_params
        
        # Wether to keep pneumonia sample or remove them
        self.keep_pneumonia = keep_pneumonia
        
        # Name of the selected train dataset
        self.dataset_name = dataset_name
        
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
        model_fname = f"{self.hyperparameters['architecture']}_{model_id}"
        
        # Base args dict
        args = { "dataset"          :   self.dataset_name, 
                 "output_dir"       :        self.dst_dir, 
                 "data_path"        :      self.data_path,
                 "keep_pneumonia"   : self.keep_pneumonia,
                 "ignore_check"     :        ignore_check,
                 "model_hash"       :            model_id, 
                 "model_filename"   :         model_fname,
                 "load_from"        :   global_model_path,
                 "max_train_steps"  :     max_train_steps,
                 "epochs_per_step"  :          num_epochs,
                 "current_epoch_num":           epoch_idx,
                 "eval_partition"   :              "test",
               }
        
        for dictionary in [self.aug_params, self.hyperparameters]:
            args.update(dictionary)
        
        command = ["python", "-m", "train_model"]
        for k, v in args.items():
            command.extend([self.get_flag_from_type(k, v), str(k), str(v)])

        # Trains model
        subprocess.Popen.wait(subprocess.Popen( command ))

        # Gets the path to the trained model
        local_model_path = os.path.join( self.dst_dir, self.dataset.name, 
                                         model_fname, f"{model_fname}.h5" )
        
        # Returns the path to the trained model and n° of train samples
        return local_model_path