import os
import json
import shutil
import subprocess
import numpy as np
import pandas as pd

# 
from utils.dataset import Dataset
from utils.custom_plots import CustomPlots
from utils.custom_models import ModelBuilder
from utils.custom_model_trainer import ModelHandler
from utils.custom_model_trainer import ModelManager
from utils.federated_callbacks import FederatedEarlyStopping
from utils.federated_callbacks import FederatedModelCheckpoint

class FederatedServer(ModelHandler):
    def __init__(self, path_dict, model_fname, model_id, val_dataset, 
                 fl_params, hyperparams, aug_params, keep_pneumonia, 
                 ignore_check):
        
        # Inherits ModelHandler's init method
        ModelHandler.__init__( self, path_dict["outputs"], model_fname, 
                               model_id )
        
        # Path to store latest version of global model
        # self.model_path is reserved for the best set of weights
        # or the final weights, if FederatedModelCheckpoint is disabled
        self.latest_model_path = os.path.join(self.model_dir, 
                                              "latest_global_model", 
                                              f"{self.model_fname}.h5")
        
        # Path to CSV file w/ val metrics for all versions of the global model
        self.history_path = os.path.join(self.model_dir, "history_dict.csv")
        
        # Prints the selected hyperparameters
        print("\nUsing the current hyperparameters for Federated Learning:")
        self.fl_params = fl_params
        self.print_dict(self.fl_params)
        
        # Prints the selected hyperparameters
        print("\nUsing the current hyperparameters for Model Training:")
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
        
        # Defines the metric used to monitor model performance
        self.monitor_var = f"avg_{self.hyperparameters['monitor']}"
        
        # Sets training callbacks for FL simulation
        self.model_checkpoint = FederatedModelCheckpoint(self.model_path, 
                                                         self.monitor_var)
        
        self.early_stopping = FederatedEarlyStopping(self.monitor_var,
                         min_delta = self.hyperparameters["early_stop_delta"],
                patience_epochs = self.hyperparameters["early_stop_patience"],
                          epochs_per_step = self.fl_params["epochs_per_step"])
        
        # Dataset used to validate models
        self.dataset = Dataset( self.data_path, name = val_dataset, 
                                keep_pneumonia = self.keep_pneumonia )
        
        # Initializes an empty client dict
        self.client_dict = {}
        
        # Initializes an empty dict to store client's training sample count
        self.num_samples_dict = {}

        return
    
    def create_global_model(self):
        
        # Gets the Filename/Folder for the current version of the global model
        latest_model_dir = os.path.dirname(self.latest_model_path)
        
        # Creates directory if needed
        if not os.path.exists(latest_model_dir):
            os.makedirs(latest_model_dir)
            
        # Creates the model
        model_builder = ModelBuilder( model_path = self.latest_model_path )
        model = model_builder( self.hyperparameters, 
                               seed = self.hyperparameters["seed"] )
        
        # Saves model config & model weights
        print(f"\nSaving global model to '{self.latest_model_path}'...")
        self.save_model(model, self.latest_model_path)
        
        return self.latest_model_path
    
    def combine_local_results( self, df_dict ):
        dfs = []
        for client_id, client_df in df_dict.items():
            # Copies the local client's results dataframe
            df = client_df.copy(deep = True)
            
            # Changes it's column's names to add client_ids
            col_prefix = f"client_{client_id}"
            new_col_names = {col: f"{col_prefix}_{col}" for col in df.columns}
            df.rename(columns = new_col_names, inplace=True)
            
            # Appends to df list in order to combine
            dfs.append(df)
        
        # Concatenates dataframes to make an updated client history_dict
        combined_df = pd.concat( dfs, axis = 1 )
        
        # Updates combined_df with min/avg/max values for each computed metric
        col_order = []
        for metric in ["loss", "val_loss", "acc", "val_acc", "f1", "val_f1"]:
            sel_cols = [f"client_{c_id}_{metric}" for c_id in df_dict.keys()]
            
            # Computes min/avg/max values for each metric
            sub_df = combined_df[sel_cols].copy(deep=True)
            combined_df[f"min_{metric}"] = sub_df.min(axis = 1)
            combined_df[f"avg_{metric}"] = sub_df.mean(axis = 1)
            combined_df[f"max_{metric}"] = sub_df.max(axis = 1)
            
            # Updates the list of columns to set its order
            metric_cols = [f"{op}_{metric}" for op in ["min", "avg", "max"]]
            col_order.extend(metric_cols)
            col_order.extend(sel_cols)
        
        # Finally, sets the order of the columns in the dataframe
        combined_df = combined_df[col_order]
            
        return combined_df
    
    def validate_global_model(self, model_path, step_idx, num_steps):
        val_df_dict = {}
        for client_id, client in self.client_dict.items():
            # Passes global model to current client
            client.get_global_model(model_path)
            
            # Evaluates model on clients train/val data
            val_results_df = client.run_validation_process(step_idx)
            
            # Appends dataframe to list
            val_df_dict[client_id] = val_results_df
        
        # Combines all obtained metrics into a single DataFrame with each
        # individual value and their min/max/average values for each step
        cross_val_df = self.combine_local_results(val_df_dict)
        
        # Formats average results as dict, and prints its values
        print(f"\n{step_idx}.0/{num_steps-1} Average Global Model Results:")
        sel_cols  = [c for c in cross_val_df.columns if "avg_" in c]
        cval_dict = {c: cross_val_df.iloc[0][c] for c in sel_cols}
        self.print_dict(cval_dict, round = True)

        # Adds a new column for the current Aggregation step
        cross_val_df.insert(0, "Step.Epoch", [f"{step_idx+1}.0"])
    
        # Updates the server's CSV file with all computed metrics
        self.update_global_history(cross_val_df)
        
        # Returns the new global model's value for the monitored metric
        monitored_val = cross_val_df.iloc[0][self.monitor_var]
        return monitored_val
    
    def load_history(self, mode = "full"):
        if os.path.exists(self.history_path):
            df = pd.read_csv( self.history_path, sep = ";" )
            
            if mode.lower() == "full":
                return df
            
            idxs = df["Step.Epoch"].to_list()
            global_idxs = [idx for idx in idxs if ".0" in str(idx)]
            
            if mode.lower() == "global":
                dst_df = df.loc[df["Step.Epoch"].isin(global_idxs)]
                
            else:
                dst_df = df.loc[~df["Step.Epoch"].isin(global_idxs)]
    
            # Resets Index
            dst_df = dst_df.reset_index(drop=True)
            return dst_df
            
        return
    
    def update_global_history(self, model_history_df):
        
        # List of history dicts
        df_list = []
        
        # Appends client's history_dict if it already exists
        if os.path.exists(self.history_path):
            server_history_df = self.load_history(mode = "full")
            df_list.append( server_history_df )
        
        # Appends current local model's to df_list
        df_list.append(model_history_df)
        
        # Concatenates dataframes to make an updated client history_dict
        updated_df = pd.concat( df_list, ignore_index = True )
        
        # Fills empty values from unselected clients by repeating their latest
        # recorded value. The local model doesn't change, nor do its metrics
        updated_df = updated_df.ffill()
        
        # Saves the dataframe as CSV
        updated_df.to_csv( self.history_path, index = False, sep = ";" )
        
        return
    
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
    
    def get_client_weights(self, selected_ids):
    
        samples = { _id: self.num_samples_dict[_id] for _id in selected_ids }
        total_samples = np.sum(list(samples.values()))
        
        weights = { _id: s / total_samples  for _id, s in samples.items() }
        
        return weights
    
    def get_max_train_steps(self, selected_ids):
        # Extracts relevant hyperparameters from hyperparameters dict
        sampling = self.hyperparameters["sampling"]
        batchsize = self.hyperparameters["batchsize"]
        
        # Lists the maximum number of training steps (batches) 
        # that each of the selected client can produce
        num_step_list = []
        for _id in selected_ids:
            client = self.client_dict[_id]
            num_step = client.dataset.get_num_steps( "train", batchsize,
                                                     sampling = sampling )
            
            num_step_list.append(num_step)
        
        # Extracts the minimum and the maximum values from that list
        min_steps = np.min(num_step_list)
        max_steps = np.max(num_step_list)
        
        # Computes the 'max_train_steps' as a value between min_steps and
        # max_steps that's closer to min_steps as 'max_steps_frac' is closer
        # to 0 and is closer to max_steps as 'max_steps_frac' is closer to 1
        steps_frac = self.fl_params["max_steps_frac"]
        xtra_steps = (max_steps - min_steps) * steps_frac
        max_train_steps = int(min_steps + xtra_steps)
        
        return max_train_steps
    
    def get_epoch_info( self, step_idx ):
        assert step_idx > 0, "\nVariable 'step_idx' must be > 0"
        
        # Retrieves total n° of epochs / n° of epochs between aggregations
        epochs_per_step = self.fl_params["epochs_per_step"]
        total_epoch_num = self.hyperparameters["num_epochs"]
        
        # Gets the number of epochs were executed so far
        epoch_idx = (step_idx - 1) * epochs_per_step
        
        # Gets the number of epochs to perform in the current step
        step_epochs = epochs_per_step
        if (epoch_idx + step_epochs) > total_epoch_num:
            # This value is corrected if it would surpass the 
            # total amount of epochs from self.hyperparameters
            step_epochs = (total_epoch_num - epoch_idx)
        
        return epoch_idx, step_epochs
    
    def train_local_models(self, step_idx, num_steps):
    
        # Selects clients to the current round
        selected_ids = self.select_clients()
        
        # Computes the weight of each client's gradients for the aggregation
        client_weights = self.get_client_weights(selected_ids)
        
        # Computes the maximum amount of training steps allowed
        max_train_steps = self.get_max_train_steps(selected_ids)
        
        # Gets the index of the current epoch and the number of epochs to
        # be performed by each local model
        current_epoch, step_num_epochs = self.get_epoch_info(step_idx)
        
        # Dict to register model paths and number of samples
        local_model_paths = {}
        local_model_results = {}
        for client_id in selected_ids:
            # Trains a local model for the current selected client
            client = self.client_dict[client_id]
            local_return_dict = client.run_train_process(step_idx, 
                                    epoch_idx = current_epoch,
                                    num_epochs = step_num_epochs, 
                                    # max_train_steps = 10,
                                    max_train_steps = max_train_steps,
                                    )
            
            # Appends the path and results to corresponding the dicts
            local_model_paths[client_id] = local_return_dict["path"]
            local_model_results[client_id] = local_return_dict["results"]
        
        # Combines all obtained metrics into a single DataFrame with each
        # individual value and their min/max/average values for each step
        cross_val_df = self.combine_local_results(local_model_results)
        
        # Formats average results as dict, and prints its values
        print(f"\n{step_idx-1}.5/{num_steps-1} Average Local Model Results:")
        sel_cols  = [c for c in cross_val_df.columns if "avg_" in c]
        cval_dict = {c: cross_val_df.iloc[-1][c] for c in sel_cols}
        self.print_dict(cval_dict, round = True)

        # Adds a new column for the current Aggregation step
        step_ticks = [f"{step_idx}.{i+1}" for i in range(len(cross_val_df))]
        cross_val_df.insert(0, "Step.Epoch", step_ticks)
    
        # Updates the server's CSV file with all computed metrics
        self.update_global_history(cross_val_df)
            
        return client_weights, local_model_paths
    
    def federated_average(self, local_model_paths, client_weights):
        
        # Gets weights from each trained model
        model_weights = []
        for local_path in local_model_paths.values():
            local_model = self.load_model(local_path)
            model_weights.append( local_model.get_weights() )
            print(f"\tLoaded weights from '{local_path}'...")
        
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
        
        if self.fl_params['aggregation'].lower() == "fed_avg":
            global_model_weights = self.federated_average( local_model_paths,
                                                           client_weights )
        
        else:
            client_weights = { k: 1 for k in client_weights.keys() }
            global_model_weights = self.federated_average( local_model_paths,
                                                           client_weights )
        
        # Loads old model and replaces weights
        model = self.load_model(self.latest_model_path)
        model.set_weights(global_model_weights)
        print("\nAggregation completed!")
        
        # Saves model config & model weights
        print(f"\nSaving global model to '{self.latest_model_path}'...")
        self.save_model(model, self.latest_model_path)
        
        return self.latest_model_path
    
    def get_num_steps(self):
        # Retrieves n° of epochs / n° of epochs per aggregation
        epochs_per_step = self.fl_params["epochs_per_step"]
        total_epoch_num = self.hyperparameters["num_epochs"]
        
        # Returns the number of training steps
        # Step 0 for global model creation and 
        # Steps 1:num_steps to global model update
        num_steps = 1 + int(np.ceil(total_epoch_num / epochs_per_step))
        return num_steps
    
    def get_client_path_dict(self):
        path_dict = { "datasets": self.data_path,
                      "outputs" : os.path.join(self.model_dir, "clients"),
                    }
        return path_dict
    
    def run_test_process( self ):
        
        # Base args dict
        args = { "output_dir"         :                 self.dst_dir, 
                 "data_path"          :               self.data_path,
                 "dataset"            :       self.dataset.orig_name, 
                 "keep_pneumonia"     :          self.keep_pneumonia,
                 "ignore_check"       :            self.ignore_check,
                 "model_hash"         :                self.model_id, 
                 "model_filename"     :             self.model_fname,
                 "eval_partition"     :                       "test",
                 "hyperparameters"    :         self.hyperparameters,
                 "data_augmentation"  :              self.aug_params,
                 "use_validation_data":                        False,
                 "verbose"            :                            0,
                 "seed"               : self.hyperparameters["seed"],
               }
        
        # Serializes args dict as JSON formatted string
        serialized_args = json.dumps(args)
        
        # 
        command = ["python", "-m", "run_model_testing", serialized_args]

        # Evaluates model
        subprocess.Popen.wait(subprocess.Popen( command ))
        return
    
    def plot_train_results(self):
        # Path to CSV file w/ val metrics for all versions of the global model
        assert os.path.exists(self.history_path), f"Can't find '{self.history_path}'..."
  
        # Object responsible for plotting
        plotter = CustomPlots(self.model_path)
        
        # Plots complete training results
        plotter.plot_fl_global_results( self.load_history(mode = "full"), 
                                        prefix = "full" )

        # Plots only results realted to Global models
        plotter.plot_fl_global_results( self.load_history(mode = "global"), 
                                        prefix = "partial" )
        
        # Dict associating client_ids its client's dataset's name
        client_dset_dict = { _id: client.dataset.name for _id, client in\
                                                 self.client_dict.items() }

        # Plots results realted to each client's model
        plotter.plot_fl_local_results( self.load_history(mode = "full"),
                                       client_dset_dict, prefix = "full")

        # Plots results realted to each client's model
        plotter.plot_fl_local_results( self.load_history(mode = "global"),
                                       client_dset_dict, prefix = "partial")
        
        return
    
    def clear_tmp_models(self):
        """ Deletes all temporary models to keep only the final 
        ModelCheckpoint in FederatedServer """
        
        # Deletes latest global model
        latest_model_dir = os.path.dirname(self.latest_model_path)
        if os.path.exists(latest_model_dir):
            shutil.rmtree(latest_model_dir, ignore_errors = False)
        
        # Also deletes all client's global/local models
        for client in self.client_dict.values():
            client.rm_model(model_type = "all")
            
        return
    
    @staticmethod
    def save_model(model, weights_path):
        
        # Gets the Folder where the model will be saved
        model_dir = os.path.dirname(weights_path)
        
        # Creates directory if needed
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Saves model configs
        json_config = model.to_json()
        config_path = weights_path.replace(".h5", ".json")
        
        # Deletes config/weight files if they already exist
        for path in [config_path, weights_path]:
            if os.path.exists(path):
                os.remove(path)

        # Saves model's config as .json file
        with open(config_path, "w") as json_file:
            json.dump( json_config, json_file, indent=4 )
        
        # Saves model's weights as .h5 file
        model.save_weights( weights_path )
        
        return

class FederatedClient():
    def __init__(self, id, path_dict, dataset_name, hyperparameters, 
                 aug_params, keep_pneumonia, ignore_check):
        
        #
        self.client_id = id
        self.client_name = f"client_{str(self.client_id).zfill(2)}"
        
        # Directory for all available datasets
        self.data_path = path_dict["datasets"]
        
        # Directory for the output trained models
        self.dst_dir = os.path.join(path_dict["outputs"], self.client_name)
        
        # Path to store latest version of this client's global model
        self.global_model_path = os.path.join(self.dst_dir, "global_model",
                                              "global_model.h5")
        
        # Path to store latest version of this client's local model
        self.local_model_path = os.path.join(self.dst_dir, "local_model",
                                             "local_model.h5")
        
        # Hyperparameters used for training
        self.hyperparameters = hyperparameters
        
        # Data augmentation parameters used for training
        self.aug_params = aug_params
        
        # Wether to keep pneumonia sample or remove them
        self.keep_pneumonia = keep_pneumonia
        
        #
        self.ignore_check = ignore_check
        
        # Builds object to handle the training dataset
        self.dataset = Dataset( self.data_path, name = dataset_name, 
                                keep_pneumonia = self.keep_pneumonia )
            
        
        # Reports client's creation
        print(f"\tCreated client #{self.client_id} w/ dataset", 
              f"'{self.dataset.name}'...")

        return
    
    def rm_model(self, model_type = "all"):
        # Deletes local and/or global models based on model_type
        # For model_type == "local",  local model will be deleted
        # For model_type == "global", global model will be deleted
        # For any other string, both models will be deleted
        path_dict = {"global": self.local_model_path,
                     "local" : self.global_model_path,}
        
        # Deletes global model 
        for key, path in path_dict.items():
            if model_type.lower() != key:
                model_dir = os.path.dirname(path)
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir, ignore_errors = False)
        return
    
    def get_global_model(self, model_path):
        assert os.path.exists(model_path), f"Can't find '{model_path}'..."
        
        # Empties global model dir before copying the model
        global_model_dir = os.path.dirname(self.global_model_path)
        self.rm_model(model_type = "global")
        os.makedirs(global_model_dir)
        
        # Gets the path for the model's configs file
        src_configs_path = model_path.replace(".h5", ".json")
        dst_configs_path = self.global_model_path.replace(".h5", ".json")
        
        # Copies the selected model's weights and configs
        dst_path_list = [self.global_model_path, dst_configs_path]
        src_path_list = [model_path, src_configs_path]
        for src_path, dst_path in zip(src_path_list, dst_path_list):
            shutil.copy2(src_path, dst_path)
        return

    def run_validation_process( self, step_idx ):
        
        # Combines client_id and communication round to make local model_id
        # The model filename combines the architecture used and the model_id
        model_id = f"v{step_idx}"
        model_fname = os.path.basename(self.global_model_path).split(".")[0]
        
        # Base args dict
        args = { "output_dir"         :                 self.dst_dir, 
                 "data_path"          :               self.data_path,
                 "dataset"            :       self.dataset.orig_name, 
                 "keep_pneumonia"     :          self.keep_pneumonia,
                 "ignore_check"       :            self.ignore_check,
                 "model_hash"         :                     model_id, 
                 "model_filename"     :                  model_fname,
                 "hyperparameters"    :         self.hyperparameters,
                 "data_augmentation"  :              self.aug_params,
                 "use_validation_data":                         True,
                 "verbose"            :                            0,
                 "seed"               : self.hyperparameters["seed"],
               }
        
        # Serializes args dict as JSON formatted string
        serialized_args = json.dumps(args)
        
        # 
        command = ["python", "-m", "run_model_testing", serialized_args]

        # Validates the model
        assert os.path.exists(self.global_model_path), " ".join(f"Can't find",
                                             f"'{self.global_model_path}'...")
        subprocess.Popen.wait(subprocess.Popen( command ))
        
        # Updates the client's dict with training/validation metrics
        global_history_df = self.load_global_history()
        
        # Updates the client's dict with training/validation metrics
        self.update_client_history_dict(global_history_df.copy(deep = True), 
                                        step_idx, from_local = False)
        
        # Returns the path to the trained model and n° of train samples
        return global_history_df
    
    def load_global_history(self):
        
        # Path to current global model's validation dict, 
        # which records model's metrics for this client's train/val data
        src_history_dir  = os.path.dirname(self.global_model_path)
        val_dict_path = os.path.join(src_history_dir, "val_results.csv")
        
        # Loads local global model's validation dict as pd.DataFrame
        validation_df = pd.read_csv( val_dict_path, sep = ";" )
        
        # Copies validation_df to extract global history_dict
        history_df = validation_df.copy(deep = True)
        
        # Conversion from columns in validation_df to the ones in history_dict
        col_dict = {}
        for metric in ["loss", "acc", "f1"]:
            col_dict[f"train_{metric}"] = metric
            col_dict[f"val_{metric}"] = f"val_{metric}"
        
        # Drops unncessary columns
        drop_cols = [c for c in history_df.columns if not c in col_dict.keys()]
        history_df.drop(columns = drop_cols, inplace = True )
        
        # Renames DataFrame's columns
        history_df.rename(columns = col_dict, inplace = True)
        
        return history_df

    def run_train_process( self, step_idx, num_epochs, epoch_idx, 
                           max_train_steps ):
        
        # Empties local model dir before training a new model
        self.rm_model(model_type = "local")
        
        # Combines client_id and communication round to make local model_id
        # The model filename combines the architecture used and the model_id
        model_id = f"{self.client_id}_v{step_idx}"
        model_fname = os.path.basename(self.local_model_path).split(".")[0]
        
        # Disables EarlyStopping for local model training
        # EarlyStopping is only triggered based on the global model's updates
        hyperparameters = self.hyperparameters.copy()
        hyperparameters["monitor"] = None
        
        # Base args dict
        args = { "output_dir"        :                 self.dst_dir, 
                 "data_path"         :               self.data_path,
                 "dataset"           :       self.dataset.orig_name, 
                 "keep_pneumonia"    :          self.keep_pneumonia,
                 "ignore_check"      :            self.ignore_check,
                 "model_hash"        :                     model_id, 
                 "model_filename"    :                  model_fname,
                 "initial_weights"   :       self.global_model_path,
                 "max_train_steps"   :              max_train_steps,
                 "epochs_per_step"   :                   num_epochs,
                 "current_epoch_num" :                    epoch_idx,
                 "hyperparameters"   :              hyperparameters,
                 "data_augmentation" :              self.aug_params,
                 "remove_unfinished" :                        False,
                 "verbose"           :                            0,
                 "seed"              :      hyperparameters["seed"],
               }
        
        # Serializes args dict as JSON formatted string
        serialized_args = json.dumps(args)
        
        # 
        command = ["python", "-m", "run_model_training", serialized_args]

        # Trains model
        subprocess.Popen.wait(subprocess.Popen( command ))
        assert os.path.exists(self.local_model_path), f"Can't find '{self.local_model_path}'..."
        
        # Updates the client's dict with training/validation metrics
        local_history_df = self.load_local_history()
        
        # Updates the client's dict with training/validation metrics
        self.update_client_history_dict(local_history_df.copy(deep = True), 
                                        step_idx, from_local = True)
        
        return_dict = { "path":    self.local_model_path,
                        "results": local_history_df,
                      }
        
        # Returns the path to the trained model and n° of train samples
        return return_dict
    
    def load_local_history(self):
        # Path to current local model's history dict, 
        # which only records this step's metrics
        src_history_dir  = os.path.dirname(self.local_model_path)
        src_history_path = os.path.join(src_history_dir, "history_dict.csv")
        
        # Loads local model's history dict as pd.DataFrame
        history_df = pd.read_csv( src_history_path, sep = ";" )
        
        return history_df
    
    def update_client_history_dict(self, model_history_df, step_idx, from_local):
        
        # Path to the client's history dict,
        # which records metrics for all steps this client has participated
        dst_history_path = os.path.join(self.dst_dir, "history_dict.csv")
        
        # List of history dicts
        df_list = []
        
        # Appends client's history_dict if it already exists
        if os.path.exists(dst_history_path):
            client_history_df = pd.read_csv( dst_history_path, sep = ";" )
            df_list.append( client_history_df )

        # Adds a new column for the current Step/Epoch
        step_list = []
        if from_local:
            for i in range(len(model_history_df)):
                step_list.append(f"{step_idx}.{i+1}")  
        else:
            step_list.append(f"{step_idx}.0")
        model_history_df.insert(0, "Step.Epoch", step_list)
        
        # Appends current local model's to df_list
        df_list.append(model_history_df)
        
        # Concatenates dataframes to make an updated client history_dict
        updated_df = pd.concat( df_list, ignore_index = True )
        
        # Saves the dataframe as CSV
        updated_df.to_csv( dst_history_path, index = False, sep = ";" )
        
        return