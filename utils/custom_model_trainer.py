import os
import warnings

# Suppresses warning messages
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import json
import time
import shutil
import hashlib
import itertools
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf

# Metrics used in model evaluation
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow_addons.metrics import F1Score

# Custom models and DataGenerator
from utils.dataset import Dataset
from utils.custom_plots import CustomPlots
from utils.custom_models import ModelBuilder
from utils.custom_generator import CustomDataGenerator
class ModelEntity():
    def __init__(self):
        return

    @staticmethod
    def print_dict(dict, round = False):
        """ Prints pairs of keys and values of a given dictionary """
        max_key_length = np.max( [len(k) for k in dict.keys()] )
        for k, v in dict.items():
            v = np.round(v, 6) if round and isinstance(v, float) else v
            print(f"\t{k.ljust(max_key_length)}: {v}")
        return

class ModelManager(ModelEntity):
    def __init__(self, path_dict, dataset_name, hyperparam_values = None, 
                 aug_params = None, fl_params = None, keep_pneumonia = False):
        
        # Checks wether to simulate Federated Learning or not
        self.federated = isinstance(fl_params, dict)
        if self.federated:
            # Prints possible values for Federated Learning parameters
            print("\nSimulating Federated Learning w/ the following values:")
            self.print_dict(fl_params)
            self.fl_params = fl_params
        
        # Checks wether hyperparameter values were provided
        if isinstance(hyperparam_values, dict):
            # Prints possible values for training hyperparameters
            print("\nList of possible hyperparameter values:")
            self.print_dict(hyperparam_values)
            self.hyperparam_values = hyperparam_values
        
        # Wether to keep pneumonia sample or remove them
        self.keep_pneumonia = keep_pneumonia
        
        # Directory for all available datasets
        self.path_dict = path_dict
        
        # Train dataset for regular training
        # Validation dataset for Federated Learning simulations
        self.dataset = Dataset(self.path_dict["datasets"], name = dataset_name,
                               keep_pneumonia = self.keep_pneumonia )
        
        # Checks wether data augmentation parameter values were provided
        self.aug_params = aug_params
        if isinstance(aug_params, dict):
            # Prints the given data augmentation parameters
            print("\nUsing the current parameters for Data Augmentation:")
            self.print_dict(self.aug_params)
        
        # Sets the codes for each script as class variables
        self.train_code = 0
        self.test_code  = 1
        self.fl_code    = 2
        return
    
    def check_trainability(self, hyperparams, list_values = True):
        assert not (hyperparams is None), "\nHyperparameters not provided..."
        assert not (self.aug_params is None), "\nAugmentation not provided..."
        if list_values:
            assert all([isinstance(v, list) for v in hyperparams.values()]),\
                "All hyperparameter values should be lists..."
        else:
            assert all([isinstance(v, tuple) for v in hyperparams.values()]),\
                "All hyperparameter values should be tuples..."
        return True

    def doGridSearch( self, shuffle = False ):
        hyperparam_dict = self.get_complete_hyperparam_dict()
        self.check_trainability(hyperparam_dict, list_values = True)
        print("\nStarting GridSearch:")

        # Checks all possible permutations from the given values
        hyperparam_permutations = list(self.product_dict(**hyperparam_dict))
        print(f"\nFound {len(hyperparam_permutations)} permutations...")
        
        if shuffle:
            np.random.shuffle(hyperparam_permutations)

        # Iterates the list of permutations
        n_permutations = len(hyperparam_permutations)
        for idx_h, hyperparameters in enumerate(hyperparam_permutations):

            # Announces the start of the training process
            current_idx = str(idx_h+1).zfill(3)
            maximum_idx = str(n_permutations).zfill(3)
            print(f"\n\n{current_idx}/{maximum_idx} Iteration of GridSearch:")
            
            if self.federated:
                # Trains model simulating Federated Learning
                self.run_process( self.fl_code, hyperparameters, 
                                  ignore_check = False )
            
            else:
                # Trains model
                self.run_process( self.train_code, hyperparameters, 
                                  ignore_check = False )
                
                # Tests model
                self.run_process( self.test_code, hyperparameters, 
                                  ignore_check = False )

        return

    def doRandomSearch( self, n_models ):
        hyperparam_dict = self.get_complete_hyperparam_dict()
        self.check_trainability(hyperparam_dict, list_values = False)
        print("\nStarting RandomSearch:")

        idx_h = 0
        while idx_h < n_models:
            hyperparameters = self.gen_random_hyperparameters(hyperparam_dict)

            # Announces the start of the training process
            print(f"\n\n#{str(idx_h+1).zfill(3)}/{str(n_models).zfill(3)} Iteration of RandomSearch:")
            
            if self.federated:
                # Trains model simulating Federated Learning
                self.run_process( self.fl_code, hyperparameters, 
                                  ignore_check = False )
            
            else:
                # Trains model
                self.run_process( self.train_code, hyperparameters, 
                                  ignore_check = False )
                
                # Tests model
                self.run_process( self.test_code, hyperparameters, 
                                  ignore_check = False )

            # Increases the number of trained models
            idx_h += 1

        return

    def doTrainFromJSON(self, json_path, copy_augmentation = True, seed = None):
        # Reads JSON file to extract hyperparameters and augmentation parameters used
        fl_params, hyperparameters, aug_params = self.json_to_hyperparam( json_path )
        
        if not fl_params is None:
            self.federated = True
        
            # Combines fl_params and hyperparameters
            complete_hyperparam_dict = fl_params.copy()
            complete_hyperparam_dict.update(hyperparameters)

        else:
            complete_hyperparam_dict = hyperparameters

        # Copies the augmentation dict used if specified
        if copy_augmentation:
            self.aug_params = aug_params
        
        # Changes the random seed used if specified
        if not seed is None:
            hyperparameters["seed"] = seed
            
        if self.federated:
            # Trains model simulating Federated Learning
            self.run_process( self.fl_code, complete_hyperparam_dict, 
                              ignore_check = True )
        
        else:
            # Trains model
            self.run_process( self.train_code, complete_hyperparam_dict, 
                              ignore_check = True )
            
            # Tests model
            self.run_process( self.test_code, complete_hyperparam_dict, 
                              ignore_check = True )

        return
            
    def run_process(self, script_code, hyperparams, ignore_check):
        assert script_code in [0, 1, 2], f"Unknown script code {script_code}"
        
        # Creates test command
        command = self.create_command(hyperparams, ignore_check, script_code)

        # Runs script as subprocess
        subprocess.Popen.wait(subprocess.Popen( command ))
        return

    def create_command(self, hyperparams, ignore_check, script_code):
        
        dst_dir, model_fname, model_id = self.get_model_path(hyperparams)
        
        fl_params, hyperparams = self.split_hyperparameters(hyperparams)
        
        # Base args dict
        args = { "output_dir"       :                    dst_dir, 
                 "data_path"        : self.path_dict["datasets"],
                 "dataset"          :     self.dataset.orig_name, 
                 "keep_pneumonia"   :        self.keep_pneumonia,
                 "ignore_check"     :               ignore_check,
                 "model_hash"       :                   model_id, 
                 "model_filename"   :                model_fname,
                 "hyperparameters"  :                hyperparams,
                 "data_augmentation":            self.aug_params,
                 "verbose"          :                          1,
                 "seed"             :        hyperparams["seed"],
               }
        
        # Matches the recieved code with one of the flags
        if script_code == self.train_code:
            # Sets the command for training process
            script = "run_model_training"
            
            # Updates args dict w/ training arguments
            args["remove_unfinished"]  =  True
            args["initial_weights"]    =  None
            args["max_train_steps"]    =  None
            args["epochs_per_step"]    =  None
            args["current_epoch_num"]  =     0
        
        elif script_code == self.test_code:
            # Sets the command for testing process
            script = "run_model_testing"
            
            # Updates args dict w/ testing arguments
            args["use_validation_data"] = False
        
        elif script_code == self.fl_code:
            # Sets the command for Federated Learning simulation
            script = "run_fl_simulation"
            
            # Updates args dict w/ Federated Learning arguments
            args["fl_params"] = fl_params
        
        # Serializes args dict as JSON formatted string
        serialized_args = json.dumps(args)
        
        # 
        command = ["python", "-m", script, serialized_args]

        return command

    def get_model_name( self, hyperparameters ):
        # Creates a single dict to store all hyperparameters
        all_param_dict = {}

        # Iterates through existing dicts
        for param_dict in [ hyperparameters, self.aug_params]:
            # Adds their keys/values to all_param_dict
            all_param_dict.update(param_dict)
        
        # Hashes the produced dict to produce an unique string for
        # this current training step
        model_id = self.dict_hash( all_param_dict ) 

        # Combines model_id with the architecture name 
        # to create the model filename
        model_fname = f"{hyperparameters['architecture']}_{model_id}"
        
        # Adds a prefix in case of federated models
        if self.federated:
            model_fname = f"fl_{model_fname}"

        return model_fname, model_id
    
    def get_model_path( self, hyperparameters ):
        
        # Gets model filename and its unique hash
        model_fname, model_id = self.get_model_name( hyperparameters )
        
        # Directory for the output trained models
        if self.federated:
            dst_dir = self.path_dict["outputs"]
        # If FL training isn't being simulated, a new subdir is added to split
        # the resulting models based on which dataset they're trained on
        else:
            dst_dir = os.path.join( self.path_dict["outputs"], 
                                    self.dataset.name )

        # Directory for the output files of the current model
        model_dir  = os.path.join( dst_dir, model_fname )
            
        # Path to CSV file with testing results
        csv_path = os.path.join( dst_dir, "training_results.csv" )
        
        # Checks if a model with the same name already exists
        # possible if a combination of hyperparameters is being retrained
        if os.path.exists(model_dir):
            
            idx = 0
            if os.path.exists(csv_path):
                # Reads CSV file to count models w/ same hash
                result_df = pd.read_csv(csv_path, sep = ";")

                # Counts the amount of models with the same hash
                idx = len(result_df[result_df["model_hash"] == model_id])
            
            # Keeps the same path / fname if there're no entries
            if idx > 0:
                # Otherwise, updates model_fname and model_dir 
                # to avoid overwritting the existant model
                model_fname = f"{model_fname}_{idx+1}"
                model_dir   = os.path.join( dst_dir, model_fname )
            
        return dst_dir, model_fname, model_id
    
    def get_complete_hyperparam_dict(self):
        # Returns None if no hyperparameters are available
        if not isinstance(self.hyperparam_values, dict):
            return None
        
        # If not running a Federated Learning Simulation
        if not self.federated:
            # Returns regular hyperparameters
            return self.hyperparam_values
        
        # Otherwise, combines fl_params and hyperparameters
        complete_hyperparam_dict = self.fl_params.copy()
        complete_hyperparam_dict.update(self.hyperparam_values)
        return complete_hyperparam_dict
    
    def split_hyperparameters(self, selected_hyperparams):
        if not self.federated:
            return None, selected_hyperparams
        
        # Splits hyperparameters into fl_params & hyperparams
        selected_fl_params = {}
        for key in self.fl_params.keys():
            selected_fl_params[key] = selected_hyperparams.pop(key)
        return selected_fl_params, selected_hyperparams
    
    @staticmethod
    def product_dict(**kwargs):
        """ Takes a dict with all possible hyperparameter values and return all possible combinations of values """
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    @staticmethod
    def gen_random_hyperparameters( hyperparam_ranges ):
        hyperparams = {}
        for key, value in hyperparam_ranges.items():

            if len(value) == 1:
                hyperparams[key] = value[0]
            
            elif value[-1] == "sample":
                n_items = len(value[0])
                item_idx = np.random.randint( 0, n_items )
                hyperparams[key] = value[0][item_idx]
                
            elif value[-1] == "int":
                hyperparams[key] = int(np.random.randint( value[0], value[1]+1 ))

            elif value[-1] == "log":
                low  = np.log10( value[0] )
                high = np.log10( value[1] )
                hyperparams[key] = float(10 ** np.random.uniform( low = low, high = high ))

            else:
                hyperparams[key] = float(np.random.uniform( low = value[0], high = value[1] ))

        return hyperparams
    
    @staticmethod
    def json_to_hyperparam( json_path ):
        assert os.path.exists( json_path ), f"Error! Couldn't find '{json_path}'..."

        # Opening JSON file
        with open( json_path ) as json_file:
            data = json.load(json_file)

        # Recovers model hyperparameters from JSON file
        fl_params = data["fl_params"]
        hyperparameters = data["hyperparameters"]
        augmentation_params = data["augmentation_params"]

        return fl_params, hyperparameters, augmentation_params

    @staticmethod
    def dict_hash( src_dict ) :
        """ MD5 hash of a dictionary.
        Based on: https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
        """
        dhash = hashlib.md5()
        encoded = json.dumps(src_dict, sort_keys=True).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

class ModelHandler(ModelEntity):
    def __init__(self, dst_dir, model_fname, model_id):

        # Relative path to where all models will be stored
        self.dst_dir = dst_dir
            
        # Path to CSV file with testing results
        self.csv_path = os.path.join( self.dst_dir, "training_results.csv" )

        # Generates model path
        self.model_id = model_id
        self.model_fname = model_fname
        self.model_dir = os.path.join(self.dst_dir, self.model_fname)
        self.model_path = os.path.join(self.model_dir, 
                                       f"{self.model_fname}.h5")
        
        params_json_name = f"params_{self.model_fname}.json"
        self.params_path = os.path.join( self.model_dir, params_json_name )
            
        # Initializes other class variables
        self.model = None
            
        return

    def prepare_model_dir(self, remove_unfinished = True):
        print("\nPreparing model dir:")

        if remove_unfinished:
            # Returns True if the csv file does not exist yet
            finished_models = []
            if os.path.exists( self.csv_path ):
                results_df = pd.read_csv(self.csv_path, sep = ";")
                finished_models = results_df["model_path"].to_list()

            # Lists all model subdirs in self.dst_dir
            all_subdirs = glob.glob(os.path.join(self.dst_dir, "*"))
            all_subdirs = sorted([p for p in all_subdirs if os.path.isdir(p)])

            # Iterates through those files to check .h5 path in the CSV
            # Which indicates that train/test process finished correctly
            for path2subdir in all_subdirs:
                model_basename = os.path.split(path2subdir)[-1]
                weights_path = os.path.join(path2subdir, 
                                            f"{model_basename}.h5")

                if (weights_path in finished_models):
                    continue

                print(f"\tDeleting '{model_basename}' subdir as its training", 
                      f"did not finish properly...")
                shutil.rmtree(path2subdir, ignore_errors=False)
        
        # Creates model_dir if needed
        if not os.path.exists(self.model_dir):
            print(f"\tCreating '{self.model_dir}' subdir to place current model's files...")
            os.makedirs(self.model_dir)
            
        return
    
    def check_step( self, ignore = False ):
        # If the ignore flag is raised, the verification is ignored
        # and a model with already used hyperparameters can be trained
        if ignore:
            return True

        # Returns True if the csv file does not exist yet
        if not os.path.exists( self.csv_path ):
            return True

        # The csv fileis read and filtered for models with the same hash
        result_df = pd.read_csv(self.csv_path, sep = ";")

        # If there are any rows with the same hash, the step is skipped
        if len( result_df[result_df["model_hash"] == self.model_id] ) > 0:
            print("\tStep already executed: Skipping...")
            return False
        
        # Otherwise returns True to execute the current step
        return True

    def prepare_model( self, hyperparameters ):
        assert not self.model is None, "There's no model to prepare..."

        # Compiles the model
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits = False)
        f1_metric = F1Score( num_classes = 1, threshold = .5, average = "micro", name = "f1" )
        if hyperparameters["optimizer"].lower() == "adam":
            self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hyperparameters["start_lr"]), 
                               loss = binary_crossentropy, metrics = ["acc", f1_metric])
        else:
            self.model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = hyperparameters["start_lr"]), 
                               loss = binary_crossentropy, metrics = ["acc", f1_metric])
        return

    @staticmethod
    def load_model( model_path ):
        config_path = model_path.replace(".h5", ".json")
        # Opening JSON file
        with open( config_path ) as json_file:
            json_config = json.load(json_file)

        # Loads model from JSON configs and H5 or Tf weights
        model = tf.keras.models.model_from_json(json_config)
        model.load_weights( model_path )
        return model
    
    def hyperparam_to_json( self, hyperparameters, aug_params, training_time, 
                            fl_params = None ):

        # Builds a dict of dicts w/ hyperparameters needed to reproduce a model
        dict_of_dicts = { "training_time"      : training_time,
                          "available_GPU"      : self.get_gpu_info(),
                          "fl_params"          : fl_params, 
                          "hyperparameters"    : hyperparameters, 
                          "augmentation_params": aug_params,
                        }

        # Saves the JSON file
        with open(self.params_path, "w") as json_file:
            json.dump( dict_of_dicts, json_file, indent=4 )

        return
    
    def load_params_json( self ):
        if not os.path.exists(self.params_path):
            print(f"Can't find '{self.params_path}'...")
            return {}
        
        # Opening JSON file
        with open( self.params_path ) as json_file:
            model_params = json.load(json_file)
        
        return model_params
    
    @staticmethod
    def ellapsed_time_as_str( seconds ):
        int_secs  = int(seconds)
        str_hours = str(int_secs // 3600).zfill(2)
        str_mins  = str((int_secs % 3600) // 60).zfill(2)
        str_secs  = str(int_secs % 60).zfill(2)
        time_str  = f"{str_hours}:{str_mins}:{str_secs}"
        return time_str

    @staticmethod
    def get_gpu_info():
        command = "nvidia-smi --query-gpu=gpu_name,memory.total --format=csv"
        gpus = subprocess.check_output( command.split() ).decode("ascii")
        gpus = gpus.split("\n")[:-1][1:]

        gpus_info = []
        for info in gpus:
            name, mem = info.split(",")
            mem = int(mem.split()[0]) / 1024
            
            gpu_info = f"{name} ({mem:.1f} GB)"
            gpus_info.append(gpu_info)
            
        return ", ".join(gpus_info)

class ModelTrainer(ModelHandler):
    def __init__(self, dst_dir, dataset, model_fname, model_id):
        
        # Inherits ModelHandler's init method
        ModelHandler.__init__( self, dst_dir, model_fname, model_id )

        # Sets the dataset used for training
        self.dataset = dataset
        self.dataset.load_dataframes()

        return
    
    def get_callback_list(self, hyperparameters):
        # List of used callbacks
        callback_list = [] 

        # Adds Model Checkpoint/Early Stopping if a monitor variable is passed
        var_list = ["val_loss", "val_acc", "val_f1"]
        if hyperparameters["monitor"] in var_list:
            
            # Sets callback_mode based on the selected monitored metric
            if "loss" in hyperparameters["monitor"].lower():
                callback_mode = "min"
            else: 
                callback_mode = "max"
            
            # Model Checkpoint
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                      filepath = self.model_path,
                                      monitor = hyperparameters["monitor"],
                                      mode = callback_mode, 
                                      save_best_only = True,
                                      save_weights_only = True, 
                                      include_optimizer = False, verbose = 1)
            callback_list.append(model_checkpoint)
            
            # Early Stopping
            if hyperparameters["early_stop_patience"] > 0:
                early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor = hyperparameters["monitor"],
                            min_delta = hyperparameters["early_stop_delta"],
                            patience = hyperparameters["early_stop_patience"],
                            mode = callback_mode, verbose = 1 )
                callback_list.append(early_stopping)
        
        # Learning Rate Scheduler
        def scheduler(epoch, lr):
                
            # Number of completed steps
            steps = (epoch + 1) // hyperparameters["lr_adjust_freq"]
            
            # Coeficient to multiply initial lr and get the new lr
            new_coef = hyperparameters["lr_adjust_frac"] ** steps
            
            # Gets the new lr value and prints the change
            new_lr = hyperparameters["start_lr"] * new_coef
            
            # Gets old LR value
            old_coef = hyperparameters["lr_adjust_frac"] ** (steps-1)
            old_lr = hyperparameters["start_lr"] * old_coef
            
            # Prints only in the epochs where the lr is changed
            if (epoch + 1) % hyperparameters["lr_adjust_freq"] == 0:
                print(f"Updating LR from '{old_lr:.3E}'to '{new_lr:.3E}'...")
                
            return new_lr
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, 
                                                                verbose = 0)
        callback_list.append(lr_scheduler)
        return callback_list

    def train_model( self, hyperparameters, aug_params, initial_epoch = 0, 
                     epochs_per_step = None, max_steps = None, 
                     load_from = None ):
        
        # Announces the dataset used for training
        print(f"\nTraining model '{self.model_fname}' on '{self.dataset.name}' dataset...")

        if load_from is None:
            # Creates the Model
            model_builder = ModelBuilder( self.model_path )
            self.model = model_builder( hyperparameters, seed = hyperparameters["seed"] )
            
        else:
            # Loads weights from a specific path
            # Used when applying Federated learning
            print(f"\nLoading model from '{load_from}'...\n")
            self.model = self.load_model( load_from )
        
        # Compiles the model
        self.prepare_model( hyperparameters )

        # Gets the list of Callbacks
        callback_list = self.get_callback_list(hyperparameters)

        # Creates train data generator
        train_datagen = CustomDataGenerator( self.dataset, "train", hyperparameters, aug_dict = aug_params, shuffle = True, 
                                             sampling = hyperparameters["sampling"], seed = hyperparameters["seed"] )

        # Creates validation data generator
        val_datagen   = CustomDataGenerator( self.dataset, "val", hyperparameters, shuffle = False )

        # Gets the number of samples and the number of batches using the current batchsize
        val_steps   = len(val_datagen)
        train_steps = len(train_datagen)
        
        # Limits the maximum training steps if necessary
        if not max_steps is None:
            # val_steps = np.min([val_steps, max_steps]) # TODO: Remove this
            train_steps = np.min([train_steps, max_steps])

        # Gets class_weights from training dataset
        class_weights = self.dataset.class_weights if hyperparameters["class_weights"] else None

        if epochs_per_step is None:
            # For regular training, each step performs all epochs at once
            num_epochs = hyperparameters["num_epochs"]
        else:
            # For Federated Learning, each step performs only a few epochs.
            num_epochs = initial_epoch + epochs_per_step

        # Measures time at the start of the training process
        init_time = time.time()

        # Fits the model
        history = self.model.fit( x = train_datagen, steps_per_epoch = train_steps, 
                                  epochs = num_epochs, initial_epoch = initial_epoch, 
                                  validation_data = val_datagen, validation_steps = val_steps, 
                                  callbacks = callback_list, class_weight = class_weights, 
                                  verbose = 1, max_queue_size = 20, workers = 1, 
                                  use_multiprocessing = False
                                )
        
        # Measures total training time
        ellapsed_time = (time.time() - init_time)
        train_time = self.ellapsed_time_as_str(ellapsed_time)

        # Saves model configs
        json_config = self.model.to_json()
        config_path = self.model_path.replace(".h5", ".json")

        with open(config_path, "w") as json_file:
            json.dump( json_config, json_file, indent=4 )
        
        # Saves last epoch's weights if model_checkpoint is disabled
        # or if training object is set to save the last epoch's weights
        if not os.path.exists(self.model_path):
            self.model.save_weights( self.model_path )

        # Extracts the dict with the training and validation values for loss and IoU during training
        history_dict = history.history
  
        # Object responsible for plotting
        print(f"\nTrained model '{self.model_fname}'...")
        plotter = CustomPlots(self.model_path)
        plotter.plot_train_results( history_dict, self.dataset.name )

        return history_dict, train_time
    
    def history_to_csv(self, history_dict):
        
        # Removes 'lr' from history_dict
        if "lr" in history_dict.keys():
            history_dict.pop("lr")

        # Converts that history_dict to a DataFrame
        model_df = pd.DataFrame.from_dict( history_dict )
        
        # Gets model's name from its path
        mdl_name = self.model_fname.split(".")[0]

        # Path to CSV file
        history_csv_path = os.path.join(self.dst_dir, mdl_name, 
                                        "history_dict.csv")

        # Creates model_dir if it doesnt already exist
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)
        
        # Saves the dataframe as CSV
        model_df.to_csv( history_csv_path, index = False, sep = ";" )
        
        return

class ModelTester(ModelHandler):
    def __init__(self, dst_dir, model_fname, model_id, dataset, 
                 dataset_list = None, use_val_data = False):
        
        # Inherits ModelHandler's init method
        ModelHandler.__init__( self, dst_dir, model_fname, model_id )

        # Sets the dataset used for training if available
        self.dataset = dataset
        self.dataset.load_dataframes()
        
        # If set for validation, only evaluates train/val partitions
        self.use_val_data = use_val_data
        
        if self.use_val_data:
            self.partitions = ["train", "val"]
            self.csv_path = os.path.join(self.model_dir, "val_results.csv")
        else:
            self.partitions = ["train", "val", "test"]

        # Sets the datasets used for cross-validation if available
        self.dataset_list = dataset_list
        if not ((self.dataset_list is None) or (self.use_val_data)):
            # Loads dataframes for the datasets in dataset_list
            for i in range(len(self.dataset_list)):
                self.dataset_list[i].load_dataframes()
        
        # Creates model_dir if needed
        assert os.path.exists(self.dst_dir), f"Can't find {self.dst_dir}"
        return
    
    def evaluate_model( self, dataset, hyperparameters, partition ):
        assert not self.model is None, "Can't find a model to evaluate..."

        # Gets the number of samples and the number of batches using the current batchsize
        num_samples = dataset.get_num_samples( partition )
        num_steps = dataset.get_num_steps(partition, 
                                          hyperparameters["batchsize"])

        # Creates data generator and gets all the labels as an array
        datagen = CustomDataGenerator( dataset, partition, hyperparameters, 
                                       shuffle = False )

        # Gets all labels in the dataframe as their corresponding class numbers to compute accuracy and f1-score
        y_true = datagen.get_labels()[:num_samples]
        orig_y_true = datagen.get_labels(use_orig_labels = True)[:num_samples]

        # TODO: Remove mock_test -------------------------
        mock_test = False
        if mock_test:
            y_hat = np.random.random( (num_samples, 1) ).astype(np.float32)
            y_pred = (y_hat > 0.5).astype(np.float32)
            loss_val = np.random.rand()
        else:
            # Computes the average loss for the current partition
            y_hat = self.model.predict( datagen, 
                                batch_size = hyperparameters["batchsize"], 
                                steps = num_steps, workers = 4, verbose = 1 )
            y_pred = (y_hat > 0.5).astype(np.float32)
            
            # Gets loss function from model.evaluate
            loss_val, _, _ = self.model.evaluate(datagen, 
                                batch_size = hyperparameters["batchsize"],
                                steps = num_steps, workers = 4, verbose = 1 )
        
        # Computes metrics using scikit-learn and keras.losses
        metrics_dict = { "loss"       : loss_val,
                         "acc"        : accuracy_score( y_true, y_pred ),
                         "f1"         : f1_score( y_true, y_pred ),
                         "auc"        : roc_auc_score( y_true, y_hat ),
                         "orig_y_true": orig_y_true,
                         "y_true"     : y_true,
                         "y_hat"      : y_hat,
                         "y_pred"     : y_pred
                       }
        
        return metrics_dict

    def test_model( self, hyperparameters ):

        # Object responsible for plotting ROC curves and Confusion Matrixes
        plotter = CustomPlots(self.model_path)

        # Loads model
        print(f"\nLoading model from '{self.model_path}'...")
        self.model = self.load_model( self.model_path )
        self.prepare_model( hyperparameters )

        # Creates a dictionary with ordered keys, but no values
        results = self.get_base_results_dict()

        # Announces the dataset used for training
        print(f"\nValidating model '{self.model_fname}' on '{self.dataset.name}' dataset...")
    
        # Evaluates each partition to fill results dict
        for partition in self.partitions:
            print(f"\n\n{partition.title()}:")
            metrics_dict = self.evaluate_model( self.dataset, hyperparameters,
                                                partition )
            
            # Extracts Class Activations, Predicted Labels and True Labels
            y_hat = metrics_dict.pop("y_hat")
            y_pred = metrics_dict.pop("y_pred")
            y_true = metrics_dict.pop("y_true")
            
            # Extracts original true labels (before class remapping)
            # to plot a more complex Confusion Matrix
            orig_y_true = metrics_dict.pop("orig_y_true")

            for metric, value in metrics_dict.items():
                # Adds the results to the result dict
                key = f"{partition}_{metric}"
                results[key] = f"{value:.4f}"

            if not self.use_val_data:
            
                # Plots 2x2 confusion matrix
                class_labels = self.dataset.classes
                plotter.plot_confusion_matrix(y_true, y_pred, 
                                              self.dataset.name, 
                                              partition, class_labels)
            
                # Plots 3x2 confusion matrix if the pneumonia 
                # samples weren't dropped
                if not self.dataset.label_remap is None:
                    plotter.plot_confusion_matrix(orig_y_true, y_pred, 
                                                  self.dataset.name, 
                                                  partition, class_labels, 
                                                  self.dataset.orig_classes)

                # Plots ROC curves
                plotter.plot_roc_curve(y_true, y_hat, self.dataset.name, 
                                       partition)

        # If there are datasets for cross-validation
        cval_dataset_names = []
        if not ((self.dataset_list is None) or (self.use_val_data)):
            cval_losses, cval_accs, cval_f1s, cval_aurocs = [], [], [], []

            for dset in self.dataset_list:
                dset_name = dset.name
                cval_dataset_names.append(dset_name)
                
                # Announces the dataset used for testing
                print(f"\nCross-Validating model '{self.model_fname}' on '{dset_name}' dataset...")

                # Evaluates dataset
                metrics_dict = self.evaluate_model( dset, hyperparameters, "test" )
            
                # Extracts Class Activations, Predicted Labels and True Labels
                y_hat = metrics_dict.pop("y_hat")
                y_pred = metrics_dict.pop("y_pred")
                y_true = metrics_dict.pop("y_true")
                
                # Extracts original true labels (before class remapping)
                # to plot a more complex Confusion Matrix
                orig_y_true = metrics_dict.pop("orig_y_true")

                # Adds to list
                cval_losses.append(metrics_dict["loss"])
                cval_accs.append(metrics_dict["acc"])
                cval_f1s.append(metrics_dict["f1"])
                cval_aurocs.append(metrics_dict["auc"])

                for metric, value in metrics_dict.items():
                    # Adds the results to the result dict
                    dname = dset_name.lower().replace(" ", "")
                    key = f"{dname}_{metric}"
                    results[key] = f"{value:.4f}"
            
                # Plots 2x2 confusion matrix
                class_labels = self.dataset.classes
                plotter.plot_confusion_matrix(y_true, y_pred, 
                                            dset_name, "test", class_labels)
            
                # Plots 3x2 confusion matrix if the pneumonia 
                # samples weren't dropped
                if not self.dataset.label_remap is None:
                    plotter.plot_confusion_matrix(orig_y_true, y_pred, 
                                            dset_name, "test", class_labels,
                                                  self.dataset.orig_classes)

                # Plots ROC curves
                plotter.plot_roc_curve(y_true, y_hat, dset_name, 
                                        "test")
                
            results["crossval_loss"] = f"{np.mean(cval_losses):.4f}"
            results["crossval_acc"]  = f"{np.mean(cval_accs):.4f}"
            results["crossval_f1"]   = f"{np.mean(cval_f1s):.4f}"
            results["crossval_auc"]  = f"{np.mean(cval_aurocs):.4f}"

        if not self.use_val_data:
            plotter.plot_test_results(results, self.dataset.name, 
                                      cval_dataset_names)

        return results
    
    def get_base_results_dict( self ):
        # Generates keys and instantiates their value as None
        results = {}
        
        for metric in ["loss", "acc", "f1", "auc"]:
            # Generates 1 entry for each metric for each partition
            for partition in self.partitions:
                key = f"{partition}_{metric}"
                results[key] = None

            # If there are datasets for cross-validation
            if not ((self.dataset_list is None) or (self.use_val_data)):
                # Adds an entry for the average value across datasets
                key = f"crossval_{metric}"
                results[key] = None

                # Also generates 1 entry for each metric for each dataset
                for dset in self.dataset_list:
                    dset_name = dset.name.lower().replace(" ", "")
                    key = f"{dset_name}_{metric}"
                    results[key] = None

        return results

    def append_to_csv( self, hyperparameters, aug_params, results ):
        
        # Combines all available information about the model in a single dict
        combined_dict = { "model_path"   : self.model_path, 
                          "model_hash"   :   self.model_id,
                        }
        
        # If testing a model, recovers training info from params.json
        if not self.use_val_data:            
            # Gets training parameters from JSON file
            model_params = self.load_params_json()
            combined_dict["available_GPU"] = model_params["available_GPU"]
            combined_dict["training_time"] = model_params["training_time"]
        
        # Updates dict with results and all relevant hyperparameters
        combined_dict.update( results )
        combined_dict.update( hyperparameters )
        combined_dict.update( aug_params )

        # Wraps values from combined_dict as lists to convert to DataFrame, 
        # tuples are converted to string
        wrapped_dict = { k: [v] if not v is None else ["None"] for k,v in combined_dict.items() }

        # Converts that dictionary to a DataFrame
        model_df = pd.DataFrame.from_dict( wrapped_dict )

        # Creates model_dir if it doesnt already exist
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)

        # If the CSV file already exists
        if os.path.exists(self.csv_path):
            # Loads the old file
            old_df = pd.read_csv( self.csv_path, sep = ";" )

            # Appends the new dataframe as an extra row
            model_df = pd.concat( [old_df, model_df], ignore_index = True )
        
        # Saves the dataframe as CSV
        model_df.to_csv( self.csv_path, index = False, sep = ";" )
        return
    