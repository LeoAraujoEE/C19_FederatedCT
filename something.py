import gc
import os
import glob
import json
import shutil
import hashlib
import itertools
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
sns.set()

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

# Metrics used in model evaluation
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow_addons as tfa
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Custom models and DataGenerator
from utils.model_training.aux.custom_plots import CustomPlotter
from utils.model_training.aux.custom_models import ClassifModelBuilder
from utils.model_training.aux.custom_generator import CustomClassifDataGen

class ModelEntity():
    def __init__(self):
        return
    
    @staticmethod
    def update_dict_values(src_dict, dst_dict):
        available_keys = [k for k in src_dict.keys()]
        for key in dst_dict.keys():
            if key in available_keys:
                dst_dict[key] = src_dict[key]
            else:
                print(f"Couldn't find key '{key}' in dict...")
        return dst_dict
    
    @staticmethod
    def get_default_hyperparams():

        # List of default values for each hyperparameter
        hyperparams = { "num_epochs":                    30,  # Total N° of training epochs
                        "batchsize":                      4,  # Minibatch size
                        "early_stop":                     7,  # Early Stopping patience
                        "input_height":                 320,  # Model's input size
                        "input_width":                  467,  # Model's input size
                        "input_depth":                    3,  # Model's input size
                        "start_lr":                    1e-3,  # Starting learning rate
                        "min_lr":                      1e-6,  # Smallest learning rate value allowed
                        "lr_adjust_frac":              0.10,  # LR adjust frac (new_lr = frac x old_lr)
                        "lr_patience":                   10,  # N° of epochs between lr adjusts
                        "monitor":                 "val_f1",  # Monitored variable for callbacks
                        "optimizer":                 "adam",  # Chosen optimizer (adam or rmsprop)
                        "l1_reg":                         0,  # Amount of L1 regularization
                        "l2_reg":                         0,  # Amount of L2 regularization
                        "top_dropout":                    0,  # Dropout for CNNs top layers (Dense)
                        "base_dropout":                   0,  # Dropout for CNNs conv layers (Conv2D)
                        "num_units":                     [],  # N° of dense units per dense layer in top
                        "pooling":                    "avg",  # Global Pooling used
                        "weights":                     None,  # Pretrained weights
                        "preprocess_func":             True,  # If pretrained net's preprocessing should be used
                        "cutout_freq":                15000,  # Max frequency on generated spectrograms
                        "architecture":   "efficientnet_b0",  # Chosen architecture
                        "unfrozen_blocks":                0,  # N° of blocks unfrozen from the start
                        "ft_block_depth":                 0,  # N° of blocks to fine tune, doesnt fine tune if < 1
                        "ft_blocks_per_step":             1,  # N° of blocks to unfreeze per fine tune step
                        "ft_epochs_per_step":             1,  # N° of train epochs per fine tune step
                        "ft_lr_frac":                   1.0,  # LR adjust frac per fine tune step
                        "seed":                          42,  # Seed for pseudorandom generators
                      } 

        return hyperparams
    
    @staticmethod
    def get_default_augmentations():
        # List of default values for data augmentation
        daug_params = { # Image Augmentation -----------------------------------------------------
                        "zoom":                    0.00,     # Max zoom in/zoom out
                        "shear":                   00.0,     # Max random shear
                        "rotation":                00.0,     # Max random rotation
                        "vertical_translation":    0.00,     # Max vertical translation
                        "horizontal_translation":  0.00,     # Max horizontal translation
                        "vertical_flip":          False,     # Allow vertical flips  
                        "horizontal_flip":        False,     # Allow horizontal flips    
                        "brightness":              0.00,     # Brightness adjustment range
                        "channel_shift":           00.0,     # Random adjustment to random channel
                        "grayscale_prob":          0.00,     # Random adjustment to random channel
                        # Audio Augmentation -----------------------------------------------------
                        "polarity_inversion_prob": 0.00,
                        "mask_xaxis_prob":         0.00,     # Prob. to mask img based on x axis (freq)
                        "mask_yaxis_prob":         0.00,     # Prob. to mask img based on y axis (time)
                        "mask_axis_max_len":       0.00,     # Max length to mask based on img size
                        "time_shift_prob":         0.00,     # Probability to shift audio in time
                        "time_stretch_prob":       0.00,     # Probability to adjust audio speed
                        "time_stretch_factor":     0.00,     # Stretch factor f, random between [1-f, 1+f]
                        "add_bg_noises_prob":      0.00,     # Probability to add background noise
                        "bg_noises_min_snr":       00.0,     # min SNR relation between sample and noise
                        "bg_noises_max_snr":       00.0,     # max SNR relation between sample and noise
                      }

        return daug_params

    @staticmethod
    def print_dict(dict, round = False):
        """ Prints pairs of keys and values of a given dictionary """
        max_key_length = np.max( [len(k) for k in dict.keys()] )
        for k, v in dict.items():
            v = np.round(v, 6) if round and isinstance(v, float) else v
            print(f"\t{k.ljust(max_key_length)}: {v}")
        return

    @staticmethod
    def get_flag_from_type(key, value):
        if type(value) is str:
            return "--str"
        if type(value) is int:
            return "--int"
        if type(value) is float:
            return "--float"
        if type(value) is bool:
            return "--bool"
        if type(value) is list:
            return "--list"
        if value is None:
            return "--none"
        raise ValueError(f"Unknown type for '{key}' == '{value}' argument...")

    @staticmethod
    def decode_val_from_flag(key, value, flag):
        allowed_flags = ["--str", "--int", "--float", "--bool", "--list", "--none"]
        assert flag in allowed_flags, f"Invalid Flag '{flag}'..."
        if flag == "--str":
            return value
        if flag == "--int":
            return int(value)
        if flag == "--float":
            return float(value)
        if flag == "--bool":
            return (value == "True")
        if flag == "--list":
            return [int(v) for v in value.split(";") if v != '']
        if flag == "--none":
            return None
        raise ValueError(f"Unknown type of flag '{flag}' for '{key}' == '{value}' argument...")
    
    @staticmethod
    def decode_args(args_list):
        args_dict = {}
        args = args_list[1:]
        for i in range(0, len(args), 3):
            flag, key, value = args[i], args[i+1], args[i+2]
            args_dict[key] = ModelEntity.decode_val_from_flag(key, value, flag)
        return args_dict

    @staticmethod
    def format_value( val ):
    # Function to format values inside a dictionary
        if isinstance(val, tuple):
            return " x ".join( [str(e) for e in val] )
        if isinstance(val, list):
            return ", ".join( [str(e) for e in val] )
        if val is None:
            return "None"
        return val

class ClassifModelManager(ModelEntity):
    def __init__(self, path2train_script, dataset_dir, hyperparam_values = None, aug_params = None):
        
        assert os.path.exists(dataset_dir), f"Provided dataset dir '{dataset_dir}' does not exist..."
        self.dataset_dir = dataset_dir

        assert os.path.exists(path2train_script), f"Provided path to train script '{dataset_dir}' does not exist..."
        self.train_script = ".".join(path2train_script.split(os.sep)).replace(".py", "")
        

        self.hyperparam_values = hyperparam_values
        if not (hyperparam_values is None):
            # Prints the possible values
            print("\nList of possible hyperparameter values:")
            self.print_dict(self.hyperparam_values)

        self.aug_params = aug_params
        if not (aug_params is None):
            # Prints the given data augmentation parameters
            print("\nUsing the current parameters for Data Augmentation:")
            self.print_dict(self.aug_params)

        return
    
    def check_trainability(self):
        assert not (self.hyperparam_values is None), "\nHyperparameter values were not provided..."
        assert not (self.aug_params is None), "\nData augmentation values were not provided..."
        return True

    def doGridSearch( self, shuffle = False ):
        self.check_trainability()
        print("\nStarting GridSearch:")

        # Prints the possible values
        print("\nList of possible hyperparameter values:")
        self.print_dict(self.hyperparam_values)

        # Checks all possible permutations from the given values
        hyperparam_permutations = list(self.product_dict(**self.hyperparam_values))
        print(f"\nA total of {len(hyperparam_permutations)} hyperparameter permutations were found.")
        
        if shuffle:
            np.random.shuffle(hyperparam_permutations)

        # Iterates the list of permutations
        n_permutations = len(hyperparam_permutations)
        for idx_h, hyperparameters in enumerate(hyperparam_permutations):

            # Announces the start of the training process
            print(f"\n\n#{str(idx_h+1).zfill(3)}/{n_permutations} Iteration of GridSearch:")
            
            # Adds augmentation parameters to selected hyperparameters
            args = {"train_dataset": self.dataset_dir, "ignore_check": False}
            train_command = self.create_command(args, hyperparameters)

            # Trains and Tests the model
            subprocess.Popen.wait(subprocess.Popen( train_command ))

        return

    def doRandomSearch( self, hyperparam_ranges, n_models ):
        self.check_trainability()
        print("\nStarting RandomSearch:")

        # Prints the possible values
        print("\nList of possible hyperparameter ranges:")
        self.print_dict(hyperparam_ranges)

        idx_h = 0
        while idx_h < n_models:
            hyperparameters = self.gen_random_hyperparameters( hyperparam_ranges )

            # Announces the start of the training process
            print("\n\n#{}/{} Iteration of RandomSearch:".format( str(idx_h+1).zfill(3), str(n_models).zfill(3) ))
            
            # Adds augmentation parameters to selected hyperparameters
            args = {"train_dataset": self.dataset_dir, "ignore_check": False}
            train_command = self.create_command(args, hyperparameters)

            # Trains and Tests the model
            subprocess.Popen.wait(subprocess.Popen( train_command ))

            # Increases the number of trained models
            idx_h += 1

        return

    def doTrainFromJSON(self, json_path, copy_augmentation = True, seed = None):
        assert os.path.exists(json_path), "\nProvided JSON couldn't be found..."

        # Reads JSON file to extract hyperparameters and augmentation parameters used
        hyperparameters, aug_params = self.json_to_hyperparam( json_path )

        # Copies the augmentation dict used if specified
        if copy_augmentation:
            self.aug_params = aug_params
        
        # Changes the random seed used if specified
        if not seed is None:
            hyperparameters["seed"] = seed
            
        # Adds augmentation parameters to selected hyperparameters
        args = {"train_dataset": self.dataset_dir, "ignore_check": True}
        train_command = self.create_command(args, hyperparameters)

        # Trains and Tests the model
        subprocess.Popen.wait(subprocess.Popen( train_command ))

        return

    def create_command(self, args, hyperparams):
        for dictionary in [self.aug_params, hyperparams]:
            args.update(dictionary)
        
        command = ["python", "-m", self.train_script]
        for k, v in args.items():
            f = self.get_flag_from_type(k, v)
            if isinstance(v, list):
                v = ";".join([str(e) for e in v])
            command.extend([f, str(k), str(v)])

        return command
    
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
                hyperparams[key] = np.random.randint( value[0], value[1]+1 )

            elif value[-1] == "log":
                low  = np.log10( value[0] + 1e-6 )
                high = np.log10( value[1] + 1e-6 )
                hyperparams[key] = 10 ** np.random.uniform( low = low, high = high ) - 1e-6

            else:
                hyperparams[key] = np.random.uniform( low = value[0], high = value[1] )

        return hyperparams
    
    @staticmethod
    def json_to_hyperparam( json_path ):

        assert os.path.exists( json_path ), "Error! Couldn't find JSON file, check 'json_path'..."

        # Opening JSON file
        with open( json_path ) as json_file:
            data = json.load(json_file)

        # Recovers model hyperparameters from JSON file
        hyperparameters     = data["hyperparameters"]
        augmentation_params = data["augmentation_params"]

        return hyperparameters, augmentation_params

class ClassifModelTrainer(ModelEntity):
    def __init__(self, dataset, dataset_list = None, bg_noise_dir = None):

        # Sets the dataset used for training
        self.dataset = dataset
        print(f"\nTraining models based on '{self.dataset.name}' dataset...")

        # Sets the datasets used for cross-validation if available
        self.dataset_list = dataset_list
        if not self.dataset_list is None:

            # Prints the names of the datasets in dataset_list
            print("\nUsing the following datasets for cross-validation:")
            for idx, cval_dataset in enumerate(self.dataset_list):
                print( "\t", str(idx+1).zfill(2), getattr( cval_dataset, "name") )
        
        else:
            print("\nNo dataset used for cross-validation:")

        # Sets the datasets used for cross-validation if available
        self.bg_noise_dir = bg_noise_dir
        if not self.bg_noise_dir is None:
            print( "\tLoading Background Noises from '{}'...".format(self.bg_noise_dir) )

        # Relative path to where the models will be stored
        self.model_dir = os.path.join( "utils", "model_training", "output", "models", self.dataset.name )
        print("\nSaving models to '{}'...".format( self.model_dir ))

        # Reset Keras state to free GPU memmory
        self.reset_keras()

        return

    def reset_keras( self ):

        sess = tf.compat.v1.keras.backend.get_session()
        tf.compat.v1.keras.backend.clear_session()
        sess.close()

        try:
            del self.model
        except:
            pass

        gc.collect()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        return

    def gen_model_name( self, hyperparameters, aug_params ):
        # Creates a single dict to store all hyperparameters
        all_param_dict = {}

        # Iterates through existing dicts
        for param_dict in [ hyperparameters, aug_params]:
            # Adds their keys/values to all_param_dict
            all_param_dict.update(param_dict)
        
        # Hashes the produced dict to produce an unique string for
        # this current training step
        formatted_dict = { k: ModelEntity.format_value(v) for k,v in all_param_dict.items() }
        model_id = self.dict_hash( formatted_dict ) 

        # Combines model_id with the architecture name to create the model filename
        model_fname = "{}_{}".format( hyperparameters["architecture"], model_id )

        # Creates the full model path
        model_path = os.path.join( self.model_dir, model_fname, model_fname+".h5" )

        idx = 1
        # while os.path.exists(model_path):
        while os.path.exists( os.path.dirname(model_path) ):
            idx += 1
            model_fname = "{}_{}_{}".format( hyperparameters["architecture"], model_id, idx )
            model_path  = os.path.join( self.model_dir, model_fname, model_fname+".h5" )

        return model_path, model_id

    def prepare_model( self, hyperparameters, mock_test = False, fine_tune_step = 0 ):

        training_lr = hyperparameters["start_lr"]
        finetune_lr = hyperparameters["min_lr"] * (hyperparameters["ft_lr_frac"] ** fine_tune_step)
        learning_rate = training_lr if fine_tune_step == 0 else finetune_lr
    
        # Compiles the modelo
        macro_f1   = tfa.metrics.F1Score( num_classes = getattr( self.dataset, "n_classes"), average = "macro", name = "f1" )
        if hyperparameters["optimizer"].lower() == "adam":
            print("\n\tCompiling model with 'Adam' optimizer...")
            self.model.compile(optimizer = tf.keras.optimizers.Adam(lr = learning_rate), 
                               loss = "categorical_crossentropy", metrics = ["acc", macro_f1])
        else:
            print("\n\tCompiling model with 'RMSprop' optimizer...")
            self.model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = learning_rate), 
                               loss = "categorical_crossentropy", metrics = ["acc", macro_f1])
        
        if mock_test:
            # Extracts the expected input shape from the model's configs
            nn_config = self.model.get_config()
            _, h, w, c = nn_config["layers"][0]["config"]["batch_input_shape"]

            print("\tMocking test...\n")
            # Creates a random input to mock an inference
            mock_data = np.array( np.random.random_sample( (hyperparameters["batchsize"], h, w, c) ), dtype = np.float32 )
            _ = self.model.predict( (mock_data / np.max(mock_data)).astype(np.float32) )
            print("\tModel ready...\n")

        return

    def train_model( self, hyperparameters, aug_params, model_path, fine_tune_step = 0, reset = True ):

        # If the model isn't being fine tuned
        if fine_tune_step == 0:
            # Announces the dataset used for training
            print("\nTraining model '{}' on '{}' dataset...".format( os.path.basename( model_path ), getattr( self.dataset, "name") ))

            # Creates and compiles the Model
            print("\tCreating model...")
            model_builder = ClassifModelBuilder( model_path = model_path, gen_fig = True )
            self.model = model_builder( hyperparameters, n_classes = getattr( self.dataset, "n_classes"), seed = hyperparameters["seed"] )
            self.prepare_model( hyperparameters, fine_tune_step = fine_tune_step )
            print("\tModel created...")

        # Loads datasets - Reloads training dataset to keep the same order of examples in each train
        print("\tLoading Datasets...")
        self.dataset.load_dataframes( reload = True )
        if not self.dataset_list is None:
            for dset in self.dataset_list:
                dset.load_dataframes( reload = False )
        print("\tLoaded Datasets...")


        # Callbacks --------------------------------------------------------------------------------------------------
        # Sets callback_mode based on the selected monitored metric
        callback_mode = "min" if "loss" in hyperparameters["monitor"].lower() else "max"
        print("\n\tMonitoring '{}' with '{}' mode...\n".format(hyperparameters["monitor"], callback_mode))

        # Model Checkpoint
        model_checkpoint  = tf.keras.callbacks.ModelCheckpoint( model_path, monitor = hyperparameters["monitor"],
                                                                mode = callback_mode, save_best_only = True,
                                                                save_weights_only = True, include_optimizer=False, 
                                                                verbose = 1 )

        # Early Stopping
        early_stopping    = tf.keras.callbacks.EarlyStopping( monitor = hyperparameters["monitor"], 
                                                              patience = hyperparameters["early_stop"], 
                                                              mode = callback_mode, verbose = 1 )
        # Learning Rate Scheduler
        reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau( monitor = hyperparameters["monitor"], 
                                                                     factor = hyperparameters["lr_adjust_frac"],
                                                                     patience = hyperparameters["lr_patience"],
                                                                     min_lr = hyperparameters["min_lr"],
                                                                     mode = callback_mode, verbose = 1 )

        # List of used callbacks
        callback_list = [ model_checkpoint, early_stopping ]
        if (fine_tune_step == 0) and (hyperparameters["optimizer"].lower() != "rmsprop"):
            callback_list.append( reduce_lr_on_plateau )
        # Callbacks --------------------------------------------------------------------------------------------------

        # Creates train data generator
        train_datagen = CustomClassifDataGen( self.dataset, "train", hyperparameters, shuffle = True, aug_dict = aug_params, 
                                              bg_noises_dir = self.bg_noise_dir, seed = hyperparameters["seed"] )

        # Creates validation data generator
        val_datagen   = CustomClassifDataGen( self.dataset, "val", hyperparameters, shuffle = False )

        # Gets the number of samples and the number of batches using the current batchsize
        val_steps   = self.dataset.get_num_steps("val", hyperparameters["batchsize"])
        train_steps = self.dataset.get_num_steps("train", hyperparameters["batchsize"])

        # Gets class_weights from training dataset
        class_weights = self.dataset.class_weights

        num_epochs = hyperparameters["num_epochs"] if not fine_tune_step > 0 else hyperparameters["ft_epochs_per_step"]

        # Fits the model
        history = self.model.fit( x = train_datagen, steps_per_epoch = train_steps, epochs = num_epochs, 
                                  validation_data = val_datagen, validation_steps = val_steps, 
                                  callbacks = callback_list, class_weight = class_weights, workers = 4, 
                                  verbose = 1
                                )

        # Extracts the dict with the training and validation values for loss and IoU during training
        history_dict = history.history

        # Reset Keras state to free GPU memmory
        if reset:
            self.reset_keras()

        return history_dict

    def load_model( self, model_path ):
        config_path = model_path.replace(".h5", ".json")
        # Opening JSON file
        with open( config_path ) as json_file:
            json_config = json.load(json_file)

        # Loads model from JSON configs and H5 or Tf weights
        self.model = tf.keras.models.model_from_json(json_config)
        self.model.load_weights( model_path )
        return

    def fine_tune_model( self, model_path, hyperparameters, aug_params, history ):

        # Announces the dataset used for training
        print("\nFine Tuning model '{}' on '{}' dataset...".format( os.path.basename( model_path ), getattr( self.dataset, "name") ))

        # Extracts the current best value from the history dict
        key = hyperparameters["monitor"].lower()
        values = history[hyperparameters["monitor"]]
        best_value = np.min(values) if "loss" in key else np.max(values)
        print("\tCurrent best value for '{}' is {:.3f}".format(key, best_value))

        # Loads and compiles the Model
        print("\tLoading model...\r")
        model_builder = ClassifModelBuilder( model_path = model_path )
        self.load_model( model_path )
        print("\tModel loaded...")

        # Gets the parameters used for fine tuning the model
        # Adjusts 'block_depth' if the given number is larger than the total of blocks in the model
        total_blocks = model_builder.get_max_block( self.model )
        block_depth  = np.min( [hyperparameters["ft_block_depth"], total_blocks] )
        step_length  = hyperparameters["ft_blocks_per_step"]
        n_steps = np.ceil( block_depth / step_length ).astype(int)

        # Creates the output dict
        ft_dict = { k: [] for k in history.keys() }
        for i in range(n_steps):
            print("\nFine Tuning - Step {}/{}".format(i+1, n_steps))
            # Adjusts 'n_unfrozen_blocks' to avoid unfreezing more than 'block_depth' blocks
            n_unfrozen_blocks = np.min( [(i+1) * step_length, block_depth] ).astype(int)
            self.model = model_builder.unfreeze_blocks( self.model, n_unfrozen_blocks )
            self.prepare_model( hyperparameters, mock_test = False, fine_tune_step = (i+1) )

            # Creates a new model to save the fine-tuning results
            new_model_path = os.path.join( os.path.dirname(model_path), "ft_"+os.path.basename(model_path) )

            # Fine tunes the model
            rst_flag = ((i+1) == n_steps)
            tmp_dict = self.train_model( hyperparameters, aug_params, new_model_path, 
                                         fine_tune_step = (i+1), reset = rst_flag )

            # Appends the values from the current step to the final fine tuning history dict
            for key in tmp_dict.keys():
                ft_dict[key] = ft_dict[key] + tmp_dict[key]
            
            # Checks if the fine-tuned model is better than the original one
            new_values = ft_dict[hyperparameters["monitor"]]
            new_best_value = np.min(new_values) if "loss" in key else np.max(new_values)
            selected_value = np.min([new_best_value, best_value]) if "loss" in key else np.max([new_best_value, best_value])

            # If it is, deletes the old model and renames the new one to the old model's name
            if selected_value == new_best_value:
                os.remove( model_path )
                os.rename( new_model_path, model_path )
        
        # In the end, also deletes the final fine-tuned model to keep only a single model
        if os.path.exists( new_model_path ):
            os.remove( new_model_path )

        return ft_dict

    def get_base_results_dict( self ):
        
        # Generates all keys and instantiates their value as None in the result dict
        # The main goal of this part is to establish the order of the keys in results
        results = {}
        for metric in ["acc", "f1", "auc"]:
            # Generates 1 entry for each metric for each partition
            for partition in ["train", "val", "test"]:
                key = "{}_{}".format(partition, metric)
                results[key] = None

            # If there are datasets for cross-validation
            if not self.dataset_list is None:

                # Adds an entry for the average value across datasets
                key = "crossval_{}".format(metric)
                results[key] = None

                # Also generates 1 entry for each metric for each dataset
                for dset in self.dataset_list:
                    key = "{}_{}".format(getattr( dset, "name").lower().replace(" ", ""), metric)
                    results[key] = None

        return results
    
    def evaluate_model( self, dataset, hyperparameters, partition ):

        # Gets the number of samples and the number of batches using the current batchsize
        num_samples = dataset.get_num_samples( partition )
        num_steps = dataset.get_num_steps(partition, hyperparameters["batchsize"])

        # Creates data generator and gets all the labels as an array
        datagen = CustomClassifDataGen( dataset, partition, hyperparameters, shuffle = False )

        # Gets all labels in the dataframe as their corresponding class numbers to compute accuracy and f1-score
        y_true = datagen.get_labels()[:num_samples]

        # Converts labels to categorical values to compute AUROC
        cat_y_true = tf.keras.utils.to_categorical( y_true, num_classes = getattr( self.dataset, "n_classes"), dtype = "float32" )

        # Computes the average loss for the current partition
        scores = self.model.predict( datagen, steps = num_steps, workers = 4, verbose = 1)
        y_pred  = np.argmax( scores, axis = -1 )

        # Computes all metrics using scikit-learn
        mean_acc   = accuracy_score( y_true, y_pred )
        mean_f1    = f1_score( y_true, y_pred, average = "macro" )
        mean_auroc = roc_auc_score( cat_y_true, scores, average = "macro", multi_class = "ovr" )

        # Computes confusion matrix using scikit-learn
        conf_matrix  = confusion_matrix( y_true, y_pred )

        return mean_acc, mean_f1, mean_auroc, conf_matrix, cat_y_true, scores

    def test_model( self, model_path, hyperparameters ):
        print("\nLoading model from '{}'...".format(model_path))

        # Loads and compiles the Model
        print("\nLoading model...")
        self.load_model( model_path )
        self.prepare_model( hyperparameters, mock_test = True )
        print("\n\tModel loaded...")
        
        # Announces the dataset used for training
        dataset_name = getattr( self.dataset, "name")
        print("\nValidating model '{}' on '{}' dataset...".format( os.path.basename(model_path), 
                                                                   dataset_name ))

        # Loads dataset's dataframes if needed
        self.dataset.load_dataframes()

        # Creates a dictionary with ordered keys, but no values
        results = self.get_base_results_dict()

        # Evaluates each partition to fill results dict
        for partition in ["train", "val", "test"]:
            acc, f1_score, auroc, conf_matrix, y_true, y_preds = self.evaluate_model( self.dataset, hyperparameters, partition )

            for metric, value in zip( ["acc", "f1", "auc"], [acc, f1_score, auroc] ):
                # Adds the results to the result dict
                key = "{}_{}".format( partition, metric )
                results[key] = "{:.4f}".format( value )

            # Plots confusion matrix
            class_labels = getattr( self.dataset, "classes")
            self.plotterObj.plot_confusion_matrix( conf_matrix, dataset_name, partition, class_labels )

            # Plots ROC curves
            self.plotterObj.plot_roc_curve( y_true, y_preds, dataset_name, partition, class_labels )

        # If there are datasets for cross-validation
        if not self.dataset_list is None:
            cval_acc_list, cval_f1_list, cval_auroc_list = [], [], []

            for dset in self.dataset_list:
                dset_name = getattr( dset, "name")
                # Announces the dataset used for testing
                print("\nCross-Validating model '{}' on '{}' dataset...".format( os.path.basename(model_path),
                                                                                 dset_name ))

                # Loads dataset's dataframes if needed
                dset.load_dataframes()

                # Evaluates dataset
                acc, f1_score, auroc, conf_matrix, y_true, y_preds = self.evaluate_model( dset, hyperparameters, "test" )

                # Adds to list
                cval_acc_list.append(acc)
                cval_f1_list.append(f1_score)
                cval_auroc_list.append(auroc)

                for metric, value in zip( ["acc", "f1", "auc"], [acc, f1_score, auroc] ):
                    # Adds the results to the result dict
                    key = "{}_{}".format(dset_name.lower().replace(" ", ""), metric)
                    results[key] = "{:.4f}".format( value )

                # Plots confusion matrix
                class_labels = getattr( dset, "classes")
                self.plotterObj.plot_confusion_matrix( conf_matrix, dset_name, "test", class_labels )

                # Plots ROC curves
                self.plotterObj.plot_roc_curve( y_true, y_preds, dset_name, "test", class_labels )
                
            results["crossval_acc"] = "{:.4f}".format( np.mean(cval_acc_list) )
            results["crossval_f1"] = "{:.4f}".format( np.mean(cval_f1_list) )
            results["crossval_auc"] = "{:.4f}".format( np.mean(cval_auroc_list) )

        # Reset Keras state to free GPU memmory
        self.reset_keras()

        return results

    def append_to_csv( self, model_path, model_id, hyperparameters, aug_params, results ):

        # Combines all available information about the model in a single dictionary
        combined_dict = { "model_path": model_path, "model_hash": model_id }
        combined_dict.update( aug_params )
        combined_dict.update( hyperparameters )
        combined_dict.update( results )

        # Wraps values from combined_dict as lists to convert to DataFrame, tuples are converted to string
        wrapped_dict = { k: [self.format_value(v)] for k,v in combined_dict.items() }

        # Converts that dictionary to a DataFrame
        model_df = pd.DataFrame.from_dict( wrapped_dict )

        # Path to CSV file
        csv_path = os.path.join( self.model_dir, "training_results.csv" )

        # Creates model_dir if it doesnt already exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # If the CSV file already exists
        if os.path.exists(csv_path):
            # Loads the old file
            old_df = pd.read_csv( csv_path, sep = ";" )

            # Appends the new dataframe as an extra row
            model_df = pd.concat( [old_df, model_df], ignore_index = True )
        
        # Saves the dataframe as CSV
        model_df.to_csv( csv_path, index = False, sep = ";" )

        return
    
    def hyperparam_to_json( self, model_path, hyperparameters, aug_params ):

        # Builds a dict of dicts w/ hyperparameters needed to reproduce a model
        dict_of_dicts = { "hyperparameters": hyperparameters, 
                         "augmentation_params": aug_params 
                        }
        
        model_dir   = os.path.dirname( model_path )
        model_fname = os.path.basename( model_path ).split(".")[0]
        json_path   = os.path.join( self.model_dir, model_fname, "params_"+model_fname+".json" )

        # Creates model_dir if it doesnt already exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Saves the JSON file
        with open(json_path, "w") as json_file:
            json.dump( dict_of_dicts, json_file, indent=4 )

        return
    
    def json_to_hyperparam( self, json_path ):

        assert os.path.exists( json_path ), "Error! Couldn't find JSON file, check 'json_path'..."

        # Opening JSON file
        with open( json_path ) as json_file:
            data = json.load(json_file)

        # Recovers model hyperparameters from JSON file
        hyperparameters     = data["hyperparameters"]
        augmentation_params = data["augmentation_params"]

        return hyperparameters, augmentation_params

    def clear_unfinished_models(self):

        # Lists all model subdirs in self.model_dir
        all_subdirs = glob.glob(os.path.join(self.model_dir, "*"))
        all_subdirs = sorted([p for p in all_subdirs if os.path.isdir(p)])

        # Iterates through those files to check for params.json
        # This file's presence indicates that train/test process finished correctly
        for path2subdir in all_subdirs:
            model_basename = os.path.split(path2subdir)[-1]
            path_to_params = os.path.join(path2subdir, f"params_{model_basename}.json")

            if os.path.exists(path_to_params):
                continue

            print(f"Deleting '{model_basename}' subdir as its training did not finish properly...")
            shutil.rmtree(path2subdir, ignore_errors=False)

        return

    def get_dicts(self, args_dict):
        # Generates hyperparameters through the default values,
        # them updates the values with the ones available in args_dict
        hyperparameters = self.get_default_hyperparams()
        hyperparameters = self.update_dict_values(args_dict, hyperparameters)
        
        # Generates data_aug_params through the default values,
        # them updates the values with the ones available in args_dict
        data_aug_params = self.get_default_augmentations()
        data_aug_params = self.update_dict_values(args_dict, data_aug_params)
        
        # Returns both dicts
        return hyperparameters, data_aug_params

    def train_test_iteration( self, args_dict ):

        # Extracts hyperparameters and parameters for data augmentation from args_dict
        hyperparameters, data_aug_params = self.get_dicts(args_dict)

        # Sets the seed for Numpy and Tensorflow
        np.random.seed( hyperparameters["seed"] )
        tf.random.set_seed( hyperparameters["seed"] )

        # Verifies if this step was already computed
        print(f"Check_Step: {self.check_step(hyperparameters, data_aug_params)}, Ignore_Check: {args_dict['ignore_check']}")
        if (self.check_step(hyperparameters, data_aug_params)) and (not args_dict["ignore_check"]):
            print("\nThis combination was already trained...")
            return

        # Removes all files in self.model_dir related to models 
        # whose training process did not finish
        self.clear_unfinished_models()

        # Generates model path and the model id
        model_path, model_id = self.gen_model_name( hyperparameters, data_aug_params )

        # Object responsible for plotting
        self.plotterObj = CustomPlotter(model_path)

        # Prints current hyperparameters and starts training
        self.print_dict( hyperparameters, round = True )
        history_dict = self.train_model( hyperparameters, data_aug_params, model_path )

        # Gets the names of the datasets used in training/testing the models
        dataset_name = self.dataset.name
        if not self.dataset_list is None:
            cval_dataset_names = [dset.name for dset in self.dataset_list]
        else:
            cval_dataset_names = None

        # Announces the end of the training process
        print(f"\nTrained model '{os.path.basename(model_path)}'. Plotting results...")
        self.plotterObj.plot_train_results( history_dict, dataset_name )

        # Fine tunes the produced model
        if hyperparameters["ft_block_depth"] > 0:
            ft_history_dict = self.fine_tune_model( model_path, hyperparameters, data_aug_params, history_dict )
            self.plotterObj.plot_train_results( ft_history_dict, dataset_name, fine_tune = True )

        # Announces the start of the testing process
        print("\nTesting model '{}'...".format( os.path.basename(model_path) ))
        results_dict = self.test_model(model_path, hyperparameters)

        print("\nPlotting test results...")
        self.plotterObj.plot_test_results(results_dict, dataset_name, cval_dataset_names)

        # Prints the results
        print("\nTest Results:")
        self.print_dict(results_dict, round = True)

        # Saves the results to a CSV file
        print("\nSaving model hyperparameters and results as CSV...")
        self.append_to_csv( model_path, model_id, hyperparameters, data_aug_params, results_dict )

        #
        print("\nSaving training hyperparameters as JSON...")
        self.hyperparam_to_json(model_path, hyperparameters, data_aug_params)

        return
    
    def check_step( self, hyperparameters, aug_params ):

        # Generates model_id
        _, model_id = self.gen_model_name( hyperparameters, aug_params )

        # Path to CSV file
        csv_path = os.path.join( self.model_dir, "training_results.csv" )

        # Returns false if the csv file does not exist yet
        if not os.path.exists( csv_path ):
            return False

        # If the csv file exists, it is read and filtered for models with the same hash
        result_df = pd.read_csv(csv_path, sep = ";")

        # If there are any rows with the same hash, the step is skipped
        if len( result_df[result_df["model_hash"] == model_id] ) > 0:
            return True
        # Otherwise returns false to execute the current step
        return False

    @staticmethod
    def dict_hash( src_dict ) :
        """ MD5 hash of a dictionary.
        Based on: https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
        """
        dhash = hashlib.md5()
        encoded = json.dumps(src_dict, sort_keys=True).encode()
        dhash.update(encoded)
        return dhash.hexdigest()
