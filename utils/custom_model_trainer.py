import gc
import os
import glob
import json
import time
import shutil
import random
import hashlib
import itertools
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

# Metrics used in model evaluation
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Custom models and DataGenerator
from utils.custom_plots import CustomPlots
from utils.custom_models import ModelBuilder
from utils.custom_generator import CustomDataGenerator

class ModelEntity():
    def __init__(self):
        return
    
    @staticmethod
    def update_dict_values(src_dict, dst_dict):
        available_keys = [k for k in src_dict.keys()]
        for key in dst_dict.keys():
            if key in available_keys:
                dst_dict[key] = src_dict[key]
        return dst_dict
    
    @staticmethod
    def get_default_hyperparams():
        # List of default values for each hyperparameter
        hyperparams = { "num_epochs":                     1,  # Total N° of training epochs
                        "batchsize":                      8,  # Minibatch size
                        "early_stop":                    50,  # Early Stopping patience
                        "input_height":                 512,  # Model's input height
                        "input_width":                  512,  # Model's input width
                        "input_channels":                 3,  # Model's input channels
                        "apply_undersampling":        False,  # Wether to apply Random Undersampling
                        "start_lr":                    1e-3,  # Starting learning rate
                        "min_lr":                      1e-5,  # Smallest learning rate value allowed
                        "lr_adjust_frac":              0.70,  # N° of epochs between lr adjusts
                        "lr_patience":                   50,  # N° of epochs between lr adjusts
                        "class_weights":              False,  # If class_weights should be used
                        "monitor":                "val_acc",  # Monitored variable for callbacks
                        "optimizer":                 "adam",  # Chosen optimizer
                        "l1_reg":                      0.00,  # Amount of L1 regularization
                        "l2_reg":                      0.00,  # Amount of L2 regularization
                        "base_dropout":                0.00,  # SpatialDropout2d between blocks in convolutional base
                        "top_dropout":                 0.00,  # Dropout between dense layers in model top
                        "augmentation":               False,  # If data augmentation should be used
                        "architecture":   "efficientnet_b0",  # Chosen architecture
                        "seed":                           1,  # Seed for pseudorandom generators
                        } 
        return hyperparams
    
    @staticmethod
    def get_default_augmentations():
        # List of default values for data augmentation
        daug_params = { "zoom_in":                     0.00,  # Max zoom in
                        "zoom_out":                    0.00,  # Max zoom out
                        "shear":                       00.0,  # Max random shear
                        "rotation":                    00.0,  # Max random rotation
                        "vertical_translation":        0.00,  # Max vertical translation
                        "horizontal_translation":      0.00,  # Max horizontal translation
                        "vertical_flip":              False,  # Allow vertical flips  
                        "horizontal_flip":            False,  # Allow horizontal flips    
                        "brightness":                  0.00,  # Brightness adjustment range
                        "channel_shift":               00.0,  # Random adjustment to random channel
                        "constant_val":                00.0,  # Constant value used to fill image
                        "fill_mode":              "constant"  # Mode used to fill image
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
            return "-s"
        if type(value) is int:
            return "-i"
        if type(value) is float:
            return "-f"
        if type(value) is bool:
            return "-b"
        if value is None:
            return "-n"
        raise ValueError(f"Unknown type for '{key}' == '{value}' argument of type '{type(value)}'...")

    @staticmethod
    def decode_val_from_flag(key, value, flag):
        assert flag in ["-s", "-i", "-f", "-b", "-n"], f"Invalid Flag '{flag}'..."
        if flag == "-s":
            return value
        if flag == "-i":
            return int(value)
        if flag == "-f":
            return float(value)
        if flag == "-b":
            return (value == "True")
        if flag == "-n":
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
    def ellapsed_time_as_str( seconds ):
        int_secs  = int(seconds)
        str_hours = str(int_secs // 3600).zfill(2)
        str_mins  = str((int_secs % 3600) // 60).zfill(2)
        str_secs  = str(int_secs % 60).zfill(2)
        time_str  = f"{str_hours}:{str_mins}:{str_secs}"
        return time_str

class ModelManager(ModelEntity):
    def __init__(self, dataset_dir, hyperparam_values = None, aug_params = None, keep_pneumonia = False):
        
        # Directory of the selected train dataset
        self.dataset_dir = dataset_dir
        
        # Wether to keep pneumonia sample or remove them
        self.keep_pneumonia = keep_pneumonia
        
        self.hyperparam_values = hyperparam_values
        if not (self.hyperparam_values is None):
            # Prints the possible values
            print("\nList of possible hyperparameter values:")
            self.print_dict(self.hyperparam_values)
        
        self.aug_params = aug_params
        if not (self.aug_params is None):
            # Prints the given data augmentation parameters
            print("\nUsing the current parameters for Data Augmentation:")
            self.print_dict(self.aug_params)

        return
    
    def check_trainability(self):
        assert not (self.hyperparam_values is None), "\nHyperparameter values were not provided..."
        assert not (self.aug_params is None), "\nData augmentation parameters were not provided..."
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
            args = { "ignore_check": False }
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
            print(f"\n\n#{str(idx_h+1).zfill(3)}/{str(n_models).zfill(3)} Iteration of RandomSearch:")
            
            # Adds augmentation parameters to selected hyperparameters
            args = { "ignore_check": False }
            train_command = self.create_command(args, hyperparameters)

            # Trains and Tests the model
            subprocess.Popen.wait(subprocess.Popen( train_command ))

            # Increases the number of trained models
            idx_h += 1

        return

    def doTrainFromJSON(self, json_path, copy_augmentation = True, seed = None):
        # Reads JSON file to extract hyperparameters and augmentation parameters used
        hyperparameters, aug_params = self.json_to_hyperparam( json_path )

        # Copies the augmentation dict used if specified
        if copy_augmentation:
            self.aug_params = aug_params
        
        # Changes the random seed used if specified
        if not seed is None:
            hyperparameters["seed"] = seed
            
        # Adds augmentation parameters to selected hyperparameters
        args = { "ignore_check": True }
        train_command = self.create_command(args, hyperparameters)

        # Trains and Tests the model
        subprocess.Popen.wait(subprocess.Popen( train_command ))

        return
    
    def doJsonSearch(self, reference_dataset, reference_metrics, seed = None):
        """ Uses results from experiments in different datasets to search for optimal hyperparameters.
        The results CSV from 'reference_dataset' is loaded and sorted by 'reference_metrics'. Then, their
        hyperparameters are loaded from the respective JSON and used to train similar models on another dataset.
            Can also be used to retrain models from the same dataset, but with a different seed.

        Args:
            reference_dataset (str): name of the dataset whose results will be used as base.
            reference_metrics (str): name of the metrics used to sort the results.
            seed (int): seed to be used during the training process
        """

        csv_path = os.path.join( "output", "models", reference_dataset, "training_results.csv" )

        # Returns if CSV file doesn't exist
        if not os.path.exists(csv_path):
            print(f"\nCouldn't find a CSV file for '{reference_dataset}' at '{csv_path}'...")
            return
        
        # Type of sorting used. Sorts in ascending order for loss and descending for others.
        if isinstance(reference_metrics, list):
            sorting  = ["loss" in metric.lower() for metric in reference_metrics]
        else:
            sorting  = ("loss" in reference_metrics.lower())
        
        # Loads the old file
        df = pd.read_csv( csv_path, sep = ";" )
        df.sort_values(by = reference_metrics, ascending = sorting, inplace = True)
        
        # Iterates through the models
        for idx, path in enumerate(df["model_path"].to_list()):
            
            # Gets the path to the corresponding json file with hyperparameters
            dirname, basename = os.path.split(path)
            json_fname = f"params_{basename.replace('.h5', '.json')}"
            json_path  = os.path.join(dirname, json_fname)

            # Announces the start of the training process
            print(f"\n\n#{str(idx+1).zfill(3)}/{str(len(df)).zfill(3)} Iteration of JSON Search:")
            
            # Trains model based on parameters from JSON
            self.doTrainFromJSON( json_path, copy_augmentation = True, seed = seed )
            
        return

    def create_command(self, args, hyperparams):
        args["train_dataset"] = self.dataset_dir
        args["keep_pneumonia"] = self.keep_pneumonia
        for dictionary in [self.aug_params, hyperparams]:
            args.update(dictionary)
        
        command = ["python", "-m", "train_model"]
        for k, v in args.items():
            command.extend([self.get_flag_from_type(k, v), str(k), str(v)])

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
        assert os.path.exists( json_path ), "Error! Couldn't find JSON file, check 'json_path'..."

        # Opening JSON file
        with open( json_path ) as json_file:
            data = json.load(json_file)

        # Recovers model hyperparameters from JSON file
        hyperparameters = data["hyperparameters"]
        augmentation_params = data["augmentation_params"]

        return hyperparameters, augmentation_params

class ModelTrainer(ModelEntity):
    def __init__(self, dataset, dataset_list = None):

        # Sets the dataset used for training
        self.dataset = dataset
        print(f"\nTraining models using '{self.dataset.name}' dataset...")

        # Sets the datasets used for cross-validation if available
        self.dataset_list = dataset_list
        if not self.dataset_list is None:
            # Prints the names of the datasets in dataset_list
            print("\nUsing the following datasets for cross-validation:")
            for idx, cval_dataset in enumerate(self.dataset_list):
                print( "\t", str(idx+1).zfill(2), cval_dataset.name )
        
        else:
            print("\nNo dataset found for cross-validation:")

        # Relative path to where the models will be stored
        self.model_dir = os.path.join( ".", "output", "models", self.dataset.name )
        print("\nSaving model to '{}'...".format( self.model_dir ))
        
        # Creates model_dir if needed
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Reset Keras state to free GPU memmory
        self.reset_keras()

        return
        
    def train_test_iteration(self, args_dict):
        # Extracts hyperparameters and parameters for data augmentation from args_dict
        hyperparameters, data_aug_params = self.get_dicts(args_dict)

        # Sets the seed for Numpy and Tensorflow
        random.seed( hyperparameters["seed"] )
        np.random.seed( hyperparameters["seed"] )
        tf.random.set_seed( hyperparameters["seed"] )
        
        if (self.check_step(hyperparameters, data_aug_params)) and (not args_dict["ignore_check"]):
            print("\tStep already executed: Skipping...")
            return

        # Removes models whose training process did not finish properly
        self.remove_unfinished()
        
        # Generates model path and the model id
        model_path, model_id = self.gen_model_name( hyperparameters, data_aug_params )

        # Object responsible for plotting
        self.plotter = CustomPlots(model_path)

        # Prints current hyperparameters and starts training
        self.print_dict( hyperparameters, round = True )
        train_start_t = time.time()
        history_dict  = self.train_model( hyperparameters, data_aug_params, model_path )
        
        # Records the total training time
        ellapsed_time = (time.time() - train_start_t)
        train_time = self.ellapsed_time_as_str(ellapsed_time)
                
        # Gets the names of the datasets used in training/testing the models
        dataset_name = self.dataset.name
        cval_dataset_names = [ dset.name for dset in self.dataset_list ]

        # Announces the end of the training process
        print(f"\nTrained model '{os.path.basename(model_path)}' in {train_time}. Plotting results...")
        self.plotter.plot_train_results( history_dict, dataset_name )

        # Announces the start of the testing process
        print("\nTesting model '{}'...".format( os.path.basename(model_path) ))
        results_dict = self.test_model( model_path, hyperparameters )
        results_dict["train_time"] = train_time

        print("\nPlotting test results...")
        self.plotter.plot_test_results( results_dict, dataset_name, cval_dataset_names )

        # Prints the results
        print("\nTest Results:")
        self.print_dict( results_dict, round = True )

        # Saves the results to a CSV file
        print("\nSaving model hyperparameters and results as CSV...")
        self.append_to_csv( model_path, model_id, hyperparameters, data_aug_params, results_dict )

        #
        print("\nSaving training hyperparameters as JSON...")
        self.hyperparam_to_json(model_path, hyperparameters, data_aug_params)

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

    def remove_unfinished(self):

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
        model_id = self.dict_hash( all_param_dict ) 

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

    def load_model( self, model_path ):
        config_path = model_path.replace(".h5", ".json")
        # Opening JSON file
        with open( config_path ) as json_file:
            json_config = json.load(json_file)

        # Loads model from JSON configs and H5 or Tf weights
        self.model = tf.keras.models.model_from_json(json_config)
        self.model.load_weights( model_path )
        return

    def prepare_model( self, hyperparameters, mock_test = False ):

        # Compiles the model
        f1_metric = tfa.metrics.F1Score( num_classes = 1, threshold = .5, average = "micro", name = "f1" )
        if hyperparameters["optimizer"].lower() == "adam":
            print("\n\tCompiling model with 'Adam' optimizer...")
            self.model.compile(optimizer = tf.keras.optimizers.Adam(lr = hyperparameters["start_lr"]), 
                               loss = "binary_crossentropy", metrics = ["acc", "AUC", f1_metric])
        else:
            print("\n\tCompiling model with 'RMSprop' optimizer...")
            self.model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = hyperparameters["start_lr"]), 
                               loss = "binary_crossentropy", metrics = ["acc", "AUC", f1_metric])
        
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

    def train_model( self, hyperparameters, aug_params, model_path, reset = True ):
        
        # Announces the dataset used for training
        print("\nTraining model '{}' on '{}' dataset...".format( os.path.basename( model_path ), self.dataset.name ))

        # Creates and compiles the Model
        print("\tCreating model...")
        model_builder = ModelBuilder( model_path = model_path )
        self.model = model_builder( hyperparameters, seed = hyperparameters["seed"] )
        self.prepare_model( hyperparameters )
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
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint( model_path, monitor = hyperparameters["monitor"],
                                                              mode = callback_mode, save_best_only = True,
                                                              save_weights_only = True, include_optimizer=False, 
                                                              verbose = 1 )

        # Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping( monitor = hyperparameters["monitor"], 
                                                           patience = hyperparameters["early_stop"], 
                                                           mode = callback_mode, verbose = 1 )
        
        # Learning Rate Scheduler
        def scheduler(epoch, lr):
            if (epoch + 1) % 10 == 0:
                print(f"[LR Scheduler] Updating LearningRate from '{lr:.3E}' to '{lr/10:.3E}'...")
                return lr / 10
            return lr
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 0)

        # List of used callbacks
        callback_list = [ model_checkpoint, early_stopping, lr_scheduler ] 
        # Callbacks --------------------------------------------------------------------------------------------------

        # Creates train data generator
        train_datagen = CustomDataGenerator( self.dataset, "train", hyperparameters, aug_dict = aug_params, shuffle = True, 
                                             undersample = hyperparameters["apply_undersampling"], seed = hyperparameters["seed"] )

        # Creates validation data generator
        val_datagen   = CustomDataGenerator( self.dataset, "val", hyperparameters, undersample = False, shuffle = False )

        # Gets the number of samples and the number of batches using the current batchsize
        val_steps   = self.dataset.get_num_steps("val", hyperparameters["batchsize"])
        train_steps = self.dataset.get_num_steps("train", hyperparameters["batchsize"])

        # Gets class_weights from training dataset
        class_weights = self.dataset.class_weights if hyperparameters["class_weights"] else None

        num_epochs = hyperparameters["num_epochs"]

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

    def get_base_results_dict( self ):
        
        # Generates all keys and instantiates their value as None in the result dict
        # The main goal of this part is to establish the order of the keys in results
        results = { "train_time": None }
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
                    key = "{}_{}".format(dset.name.lower().replace(" ", ""), metric)
                    results[key] = None

        return results

    def evaluate_model( self, dataset, hyperparameters, partition ):

        # Gets the number of samples and the number of batches using the current batchsize
        num_samples = dataset.get_num_samples( partition )
        num_steps = dataset.get_num_steps(partition, hyperparameters["batchsize"])

        # Creates data generator and gets all the labels as an array
        datagen = CustomDataGenerator( dataset, partition, hyperparameters, shuffle = False, undersample = False )

        # Gets all labels in the dataframe as their corresponding class numbers to compute accuracy and f1-score
        y_true = datagen.get_labels()[:num_samples]
        fnames = datagen.get_fnames()[:num_samples]

        # Computes the average loss for the current partition
        scores = self.model.predict( datagen, batch_size = hyperparameters["batchsize"], 
                                     steps = num_steps, workers = 4, verbose = 1 )
        y_pred  = (scores > 0.5).astype(np.float32)

        # Computes all metrics using scikit-learn
        mean_acc   = accuracy_score( y_true, y_pred )
        mean_f1    = f1_score( y_true, y_pred )
        mean_auroc = roc_auc_score( y_true, scores )

        # Computes confusion matrix using scikit-learn
        conf_matrix  = confusion_matrix( y_true, y_pred )

        return mean_acc, mean_f1, mean_auroc, conf_matrix, y_true, scores

    def test_model( self, model_path, hyperparameters ):
        print("\nLoading model from '{}'...".format(model_path))

        ###
        print("\nLoading model...")
        self.load_model( model_path )
        self.prepare_model( hyperparameters, mock_test = True )
        print("\n\tModel loaded...")
        ###
        
        # Announces the dataset used for training
        dataset_name = self.dataset.name
        print("\nValidating model '{}' on '{}' dataset...".format( os.path.basename(model_path), 
                                                                   dataset_name ))

        # Loads dataset's dataframes if needed
        self.dataset.load_dataframes()

        # Creates a dictionary with ordered keys, but no values
        results = self.get_base_results_dict()

        # Evaluates each partition to fill results dict
        for partition in ["train", "val", "test"]:
            print("\n\n{}:".format(partition.title()))
            acc, f1_score, auroc, conf_matrix, y_true, y_preds = self.evaluate_model( self.dataset, hyperparameters, partition )

            for metric, value in zip( ["acc", "f1", "auc"], [acc, f1_score, auroc] ):
                # Adds the results to the result dict
                key = "{}_{}".format( partition, metric )
                results[key] = "{:.4f}".format( value )

            # Plots confusion matrix
            class_labels = self.dataset.classes
            self.plotter.plot_confusion_matrix( conf_matrix, dataset_name, partition, class_labels )

            # Plots ROC curves TODO: fix this
            self.plotter.plot_roc_curve( y_true, y_preds, dataset_name, partition, class_labels )

        # If there are datasets for cross-validation
        if not self.dataset_list is None:
            cval_acc_list, cval_f1_list, cval_auroc_list = [], [], []

            for dset in self.dataset_list:
                dset_name = dset.name
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
                class_labels = dset.classes
                self.plotter.plot_confusion_matrix( conf_matrix, dset_name, "test", class_labels )

                # Plots ROC curves TODO: fix this
                self.plotter.plot_roc_curve( y_true, y_preds, dset_name, "test", class_labels )
                
            results["crossval_acc"] = "{:.4f}".format( np.mean(cval_acc_list) )
            results["crossval_f1"] = "{:.4f}".format( np.mean(cval_f1_list) )
            results["crossval_auc"] = "{:.4f}".format( np.mean(cval_auroc_list) )

        # Reset Keras state to free GPU memmory
        self.reset_keras()

        return results

    def append_to_csv( self, model_path, model_id, hyperparameters, aug_params, results ):

        # Combines all available information about the model in a single dictionary
        combined_dict = { "model_path": model_path, "model_hash": model_id }
        combined_dict.update( results )
        combined_dict.update( hyperparameters )
        combined_dict.update( aug_params )

        # Wraps values from combined_dict as lists to convert to DataFrame, tuples are converted to string
        wrapped_dict = { k: [v] if not v is None else ["None"] for k,v in combined_dict.items() }

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
                          "augmentation_params": aug_params }
        
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