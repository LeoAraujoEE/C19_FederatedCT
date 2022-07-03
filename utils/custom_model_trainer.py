import gc
import os
import json
import random
import hashlib
import itertools
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
sns.set()

# Metrics used in model evaluation
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Custom models and DataGenerator
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
                        "input_height":                 256,  # Model's input height
                        "input_width":                  256,  # Model's input width
                        "input_channels":                 3,  # Model's input channels
                        "apply_undersampling":        False,  # Wether to apply Random Undersampling
                        "start_lr":                    1e-3,  # Starting learning rate
                        "min_lr":                      1e-5,  # Smallest learning rate value allowed
                        "lr_adjust_frac":              0.70,  # N° of epochs between lr adjusts
                        "lr_patience":                    4,  # N° of epochs between lr adjusts
                        "class_weights":              False,  # If class_weights should be used
                        "preprocess_func":            False,  # If keras preprocess_functions should be used
                        "monitor":                "val_acc",  # Monitored variable for callbacks
                        "optimizer":                 "adam",  # Chosen optimizer
                        "l1_reg":                       0.0,  # Amount of L1 regularization
                        "l2_reg":                       0.0,  # Amount of L2 regularization
                        "dropout":                      0.0,  # Dropout for layers in skip connections
                        "augmentation":               False,  # If data augmentation should be used
                        "pooling":                    "avg",  # Global Pooling used
                        "weights":                     None,  # Pretrained weights
                        "architecture":   "efficientnet_b0",  # Chosen architecture
                        "seed":                           1,  # Seed for pseudorandom generators
                        } 
        return hyperparams
    
    @staticmethod
    def get_default_augmentations():
        # List of default values for data augmentation
        daug_params = { "zoom":                        0.00,  # Max zoom in/zoom out
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
        raise ValueError(f"Unknown type for '{key}' == '{value}' argument...")

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

class ModelManager(ModelEntity):
    def __init__(self, dataset_dir, hyperparam_values, aug_params):
        
        self.dataset_dir = dataset_dir
        
        # Prints the possible values
        print("\nList of possible hyperparameter values:")
        self.hyperparam_values = hyperparam_values
        self.print_dict(self.hyperparam_values)

        # Prints the given data augmentation parameters
        print("\nUsing the current parameters for Data Augmentation:")
        self.aug_params = aug_params
        self.print_dict(self.aug_params)

        return

    def doGridSearch( self, shuffle = False ):
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
            
            if idx_h == 3:
                break

            # # Verifies if this step was already computed
            # if self.check_step( hyperparameters ):
            #     # If so, skips to the next step...
            #     continue

            # Announces the start of the training process
            print(f"\n\n#{str(idx_h+1).zfill(3)}/{n_permutations} Iteration of GridSearch:")
            
            # Adds augmentation parameters to selected hyperparameters
            args = {"train_dataset": self.dataset_dir}
            for dictionary in [self.aug_params, hyperparameters]:
                args.update(dictionary)
            
            command = ["python", "-m", "train_model"]
            for k, v in args.items():
                command.extend([self.get_flag_from_type(k, v), str(k), str(v)])

            # Trains and Tests the model
            subprocess.Popen.wait(subprocess.Popen( command ))

        return

    def doRandomSearch( self, hyperparam_ranges, n_models ):
        print("\nStarting RandomSearch:")

        # Prints the possible values
        print("\nList of possible hyperparameter ranges:")
        self.print_dict(hyperparam_ranges)

        idx_h = 0
        while idx_h < n_models:
            hyperparameters = self.gen_random_hyperparameters( hyperparam_ranges )

            # # Verifies if this step was already computed
            # if self.check_step( hyperparameters ):
            #     # If so, skips to the next step...
            #     continue

            # Announces the start of the training process
            print("\n\n#{}/{} Iteration of RandomSearch:".format( str(idx_h+1).zfill(3), str(n_models).zfill(3) ))

            # Trains and Tests the model
            # self.train_test_iteration( hyperparameters )
            self.print_dict(hyperparameters)

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

        # Retrains the model
        # self.train_test_iteration( hyperparameters )
        self.print_dict(hyperparameters)

        return
    
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
                hyperparams[key] = np.random.choice( value[0] )
                
            elif value[-1] == "int":
                hyperparams[key] = np.random.randint( value[0], value[1]+1 )

            elif value[-1] == "log":
                low  = np.log10( value[0] )
                high = np.log10( value[1] )
                hyperparams[key] = 10 ** np.random.uniform( low = low, high = high )

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
        
        # Generates model path and the model id
        model_path, model_id = self.gen_model_name( hyperparameters, data_aug_params )

        # Prints current hyperparameters and starts training
        self.print_dict( hyperparameters, round = True )
        history_dict = self.train_model( hyperparameters, data_aug_params, model_path )
                
        # Gets the names of the datasets used in training/testing the models
        dataset_name = self.dataset.name
        cval_dataset_names = [ dset.name for dset in self.dataset_list ]

        # Announces the end of the training process
        print("\nTrained model '{}'. Plotting results...".format( os.path.basename(model_path) ))
        self.plot_train_results( model_path, history_dict, dataset_name )

        # Announces the start of the testing process
        print("\nTesting model '{}'...".format( os.path.basename(model_path) ))
        results_dict = self.test_model( model_path, hyperparameters )

        print("\nPlotting test results...")
        self.plot_test_results( model_path, results_dict, dataset_name, cval_dataset_names )

        # Prints the results
        print("\nTest Results:")
        self.print_dict( results_dict, round = True )

        # Saves the results to a CSV file
        print("\nSaving model hyperparameters and results as CSV...")
        self.append_to_csv( model_path, model_id, hyperparameters, data_aug_params, results_dict )

        #
        print("\nSaving training hyperparameters as JSON...")
        self.hyperparam_to_json(model_path, hyperparameters, data_aug_params)

        # TODO: Retirar isso aqui
        # self.verificar_acc( model_path, hyperparameters )

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
        if hyperparameters["optimizer"].lower() != "rmsprop":
            callback_list.append( reduce_lr_on_plateau )
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
        # y_pred  = np.argmax( scores, axis = -1 )
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
            self.plot_confusion_matrix( model_path, conf_matrix, dataset_name, partition, class_labels )

            # Plots ROC curves TODO: fix this
            self.plot_roc_curve( model_path, y_true, y_preds, dataset_name, partition, class_labels )

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
                self.plot_confusion_matrix( model_path, conf_matrix, dset_name, "test", class_labels )

                # Plots ROC curves TODO: fix this
                self.plot_roc_curve( model_path, y_true, y_preds, dset_name, "test", class_labels )
                
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

        # Function to convert tuples to strings
        def tuple_as_str( tpl ):
            return " x ".join( [str(e) for e in tpl] )

        # Wraps values from combined_dict as lists to convert to DataFrame, tuples are converted to string
        wrapped_dict = { k: [v] if not isinstance(v, tuple) else [tuple_as_str(v)] for k,v in combined_dict.items() }

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
    
    def check_step( self, hyperparameters ):

        # Generates model_id
        _, model_id = self.gen_model_name( hyperparameters )

        # Path to CSV file
        csv_path = os.path.join( self.model_dir, "training_results.csv" )

        # Returns false if the csv file does not exist yet
        if not os.path.exists( csv_path ):
            return False

        # If the csv file exists, it is read and filtered for models with the same hash
        result_df = pd.read_csv(csv_path, sep = ";")

        # If there are any rows with the same hash, the step is skipped
        if len( result_df[result_df["model_hash"] == model_id] ) > 0:
            print("\tStep already executed: Skipping...")
            return True
        # Otherwise returns false to execute the current step
        return False

    def verificar_acc( self, model_path, hyperparameters ):
        print("\nLoading model from '{}'...".format(model_path))

        # Loads model
        print("\nLoading model...")
        self.load_model(model_path)
        self.prepare_model( hyperparameters, mock_test = True )
        print("\nModel loaded...")

        # Pega infos do dataset
        self.dataset.load_dataframes()
        df = self.dataset.get_dataframe("val")
        i_dir = self.dataset.get_relative_path()
        i_col = self.dataset.input_col
        o_col = self.dataset.output_col

        nn_config = self.model.get_config()
        _, h, w, c = nn_config["layers"][0]["config"]["batch_input_shape"]

        # Identifies each individual class from given labels
        unq_labels = np.unique( df[o_col].to_list() )
        label2class_dict = { label: clss for clss, label in enumerate(unq_labels) }
        class2label_dict = { clss: label for clss, label in enumerate(unq_labels) }

        erros = 0
        acertos = 0

        # itera o dataframe pra pegar preds e labels
        preds  = []
        labels = []
        scores = []
        for idx, row in df.iterrows():

            path  = row[i_col]
            label = label2class_dict[ row[o_col] ]
            preprocess_func = CustomDataGenerator.get_preprocessing_function(hyperparameters)

            # Loads input images and sets them to inputs array
            img = tf.keras.preprocessing.image.load_img( os.path.join( i_dir, path ), color_mode = "rgb", target_size = (h, w) )
            img = tf.keras.preprocessing.image.img_to_array(img).reshape( (1, h, w, c) )
            img = preprocess_func( img ).astype( np.float32 )

            score = np.squeeze(self.model( img, training = False ))
            pred  = np.argmax( score )
                

            preds.append( pred )
            labels.append( label )
            scores.append( score )

            if pred == label:
                acertos += 1
            else:
                erros += 1

            mean_acc   = 100 * (acertos / (acertos + erros))
            mean_f1    = f1_score( labels, preds, average = "macro" )
            print("{}/{} rows - Pred {} - Label {} - Acc. {:.1f} - F1 {:.3f}".format( str(idx+1).zfill(5), len(df), 
                                                                                      str(pred).zfill(2), str(label).zfill(2), 
                                                                                      mean_acc, mean_f1 ), end = "\r")
            
        score_array = np.array( scores )
        label_array = tf.keras.utils.to_categorical( np.array( labels ), num_classes = self.dataset.n_classes, dtype = "float32" )
        mean_auroc = roc_auc_score( label_array, score_array, average = "macro", multi_class = "ovr" )
        print("\n[Verif.] Accuracy: {:.2f} - F1-Score: {:.5f} - AUROC: {:.5f}".format( mean_acc, mean_f1, mean_auroc ))

        return
    
    @staticmethod
    def plot_confusion_matrix( path, conf_matrix, dataset_name, partition, labels ):
        # Defines the path to the plot file inside the model's directory
        plot_dir = os.path.join( os.path.dirname(path), "plots", "4.Confusion Matrix" )
        cm_fname = "cm_{}_{}.png".format( dataset_name, partition )
        cm_fpath = os.path.join( plot_dir, cm_fname )
        
        # Creates the plot directory if needed
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Normalizes the confusion matriz by dividing each row for its sum, each 
        # element is divided by the total amount of true samples for the true class
        n_rows, n_cols = conf_matrix.shape[:2]
        total_true_counts = np.sum( conf_matrix, axis = 1 ).reshape( n_rows, 1 )
        normalized_cm = conf_matrix / total_true_counts

        # Prepares annotations (counts and percentages) for the plot function
        cm_counts = ["{0:0.0f}".format(c) for c in conf_matrix.flatten()]
        cm_percts = ["{0:.1%}".format(p) for p in normalized_cm.flatten()]
        cm_annots = ["{}\n{}".format(c, p) for c, p in zip(cm_counts, cm_percts)]
        cm_annots = np.asarray(cm_annots).reshape( n_rows, n_cols )

        # Plots the confusion matrix as a heatmap
        plt.ioff()
        plt_h, plt_w = int(2*n_rows), int(2*n_cols)
        fig = plt.figure( figsize = (plt_h, plt_w) )
        blues_cmap = sns.color_palette("Blues", as_cmap=True)
        ax  = sns.heatmap( normalized_cm, annot = cm_annots, fmt="", cmap = blues_cmap, cbar = True, 
                           xticklabels = labels, yticklabels = labels, vmin = 0, vmax = 1)

        # Adds extra information on each label and a title
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        if partition.lower() == "test":
            ax.set_title("{} Dataset".format(dataset_name.title()))
        else:
            ax.set_title("{} Dataset ({})".format(dataset_name.title(), partition.title()))

        # Saves & closes figure
        plt.sca( ax )
        plt.savefig( cm_fpath, dpi = 100, bbox_inches = "tight" )
        plt.close( fig )

        return

    @staticmethod
    def plot_roc_curve( path, true_labels, scores, dataset_name, partition, labels ):
        
        # Defines the path to the plot file inside the model's directory
        plot_dir  = os.path.join( os.path.dirname(path), "plots", "3.ROC Curves" )
        roc_fname = "roc_{}_{}.png".format( dataset_name, partition )
        roc_fpath = os.path.join( plot_dir, roc_fname )
        
        # Creates the plot directory if needed
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(true_labels, scores)
        roc_auc  = auc(fpr, tpr)

        # Plots the confusion matrix as a heatmap
        plt.ioff()
        fig = plt.figure( figsize = (8, 12) )
        ax = plt.gca()

        # Plots ROC Curve for COVID-19
        ax.plot( fpr, tpr, color = "blue", lw = 2, label = "COVID-19 (AUROC = {:.4f})".format(roc_auc))

        # Draws a reference line for a useless classifier in each plot
        ax.plot( [0, 1], [0, 1], "k--", lw = 2 )

        # Sets the limits for each axis
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # Names each axis
        ax.set_ylabel("True Positive Rate")
        ax.set_xlabel("False Positive Rate")
        ax.legend(loc = "lower right")

        # Sets the plot title
        if partition.lower() == "test":
            ax.set_title("ROC Curves: {}".format(dataset_name.title()))
        else:
            ax.set_title("ROC Curves: {} Dataset ({})".format(dataset_name.title(), partition.title()))

        # Makes a single legend for both plots, the legend has a column for each 20 classes
        # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # ax.legend(lines, labels, bbox_to_anchor = (1., 1.))

        # Saves & closes figure
        fig.savefig( roc_fpath, dpi = 100, bbox_inches = "tight" )
        plt.close( fig )

        return
        
    @staticmethod
    def old_plot_roc_curve( path, cat_true_labels, cat_preds, dataset_name, partition, labels ):
        # Defines the path to the plot file inside the model's directory
        plot_dir  = os.path.join( os.path.dirname(path), "plots", "3.ROC Curves" )
        roc_fname = "roc_{}_{}.png".format( dataset_name, partition )
        roc_fpath = os.path.join( plot_dir, roc_fname )
        
        # Creates the plot directory if needed
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        n_samples, n_classes = cat_true_labels.shape

        # -----------------------------------------------------------------------------------------
        # -- Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html --
        # -----------------------------------------------------------------------------------------
        from sklearn.metrics import roc_curve, auc

        # Compute ROC curve and ROC area for each class
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(cat_true_labels[:, i], cat_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(cat_true_labels.ravel(), cat_preds.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # -----------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------

        # Plots the confusion matrix as a heatmap
        plt.ioff()
        fig, axes = plt.subplots(1, 2, squeeze = False, figsize = (16, 12))
        axes = axes.flat

        # Color palette
        colors = sns.color_palette(cc.glasbey, n_colors = n_classes)

        # Plots ROC Curves for individual classes in the first plot
        for c, label in enumerate(labels):
            axes[0].plot( fpr[c], tpr[c], color = colors[c], lw=2, 
                          label="{}: {} (AUROC = {:.4f})".format(c, label, roc_auc[c]))
        
        # Plots Micro Average ROC Curves for in the second plot
        # axes[1].plot( fpr["micro"], tpr["micro"], color = "b", lw=2, 
        #               label="Micro Average (AUROC = {:.4f})".format(roc_auc["micro"]))
        
        # Plots Macro Average ROC Curves for in the second plot
        axes[1].plot( fpr["macro"], tpr["macro"], color = "r", lw=2, 
                      label="Macro Average (AUROC = {:.4f})".format(roc_auc["macro"]))

        # Draws a reference line for a useless classifier in each plot
        axes[0].plot( [0, 1], [0, 1], "k--", lw = 2 )
        axes[1].plot( [0, 1], [0, 1], "k--", lw = 2 )

        for i in range(2):

            # Sets the limits for each axis
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])

            # Names each axis
            axes[i].set_ylabel("True Positive Rate")
            axes[i].set_xlabel("False Positive Rate")
            #axes[i].legend(loc = "lower right")

            # Sets the plot title
            base_title = "Class " if i == 0 else "Average "
            if partition.lower() == "test":
                axes[i].set_title(base_title + "ROC Curves: {}".format(dataset_name.title()))
            else:
                axes[i].set_title(base_title + "ROC Curves: {} Dataset ({})".format(dataset_name.title(), 
                                                                                    partition.title()))

        # Makes a single legend for both plots, the legend has a column for each 20 classes
        n_legend_cols = 1 + (n_classes // 20)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        axes[i].legend(lines, labels, ncol = n_legend_cols, bbox_to_anchor = (1., 1.))

        # Saves & closes figure
        fig.savefig( roc_fpath, dpi = 100, bbox_inches = "tight" )
        plt.close( fig )

        return

    @staticmethod
    def plot_train_results( model_path, history, dataset_name, fine_tune = False, figsize = (6, 9) ):

        # Defines the path to the plot directory inside the model's directory
        plot_dir  = os.path.join( os.path.dirname(model_path), "plots", "1.Training Results" )
        
        # Creates the plot directory if needed
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        for metric in [ "loss", "acc", "f1", "auc" ]:

            # Defines the plot name
            prefix = "ft_" if fine_tune else ""
            plot_name = prefix+"{}_{}.png".format( metric, dataset_name )
            plot_path = os.path.join( plot_dir, plot_name )

            # Extracts values from history dict
            val_values = history["val_"+metric]
            train_values = history[metric]
            all_values = list(val_values) + list(train_values)

            # Epochs
            epochs = range(1, len(train_values) + 1)

            # Plots the results
            plt.ioff()
            fig = plt.figure( figsize = figsize )

            prefix = "Fine Tuning - " if fine_tune else ""
            plt.plot( epochs, train_values, "r", label = "Training" )
            plt.plot( epochs, val_values, "b", label = "Validation" )
            plt.title( prefix+metric.title()+" per Epoch", fontsize = 24 )
            plt.xlabel( "Epochs", fontsize = 20 )
            plt.xticks( fontsize = 16 )
            plt.ylabel( metric.title(), fontsize = 20 )

            if metric != "loss":
                plt.legend( loc = "lower right", fontsize = 20 )
                plt.ylim( (np.min( [0.7, .95*np.min( all_values )] ), 1.0) )
            else:
                plt.legend( loc = "upper right", fontsize = 20 )
                plt.ylim( (0, np.min( [1.5, np.max( all_values )] )) )


            # Saves & closes figure
            fig.savefig( plot_path, dpi = 100, bbox_inches = "tight" )
            plt.close( fig )

        return

    @staticmethod
    def plot_test_results( model_path, results, dataset_name, cval_dataset_names = None, figsize = (16, 12) ):

        # Defines the path to the plot directory inside the model's directory
        plot_dir  = os.path.join( os.path.dirname(model_path), "plots", "2.Test Results" )
        
        # Creates the plot directory if needed
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        for metric in [ "acc", "f1", "auc" ]:

            # Defines the plot name
            plot_name = "{}_{}.png".format( metric, dataset_name )
            plot_path = os.path.join( plot_dir, plot_name )

            fig, axes = plt.subplots(1, 2, squeeze=False, figsize=figsize)

            # Plot 1 - Final values for Train, Validation and Test
            labels = [] # Labels for the x-axis in the bar plot
            values = [] # Values for the y-axis in the bar plot
            for partition in ["train", "val", "test"]:
                # Key to result dict
                key = "{}_{}".format(partition, metric)
            
                # Pair of label & value
                labels.append( partition.title() )
                values.append( float(results[key]) )

            # Color palette
            colors = list(sns.color_palette(cc.glasbey, n_colors = len(labels)))

            plt.sca(axes.flat[0])
            bars = plt.bar( labels, values, color = colors, width = 0.4 )
            plt.bar_label(bars)
            plt.title("{} per Partition".format(metric.title()), fontsize = 24)
            plt.xlabel("Partition", fontsize = 20)
            plt.xticks(fontsize = 16, rotation = 45)
            plt.ylabel(metric.title(), fontsize = 20)
            plt.ylim( (np.min( [0.7, .95*np.min( values )] ), 1.0) )

            # Plot 2 - Final values for Test & Cross-Val Datasets
            labels = [] # Labels for the x-axis in the bar plot
            values = [] # Values for the y-axis in the bar plot
            dset_names = ["test"] if cval_dataset_names is None else ["test"]+cval_dataset_names
            for name in (dset_names):
                # Key to result dict
                key = "{}_{}".format(name.lower().replace(" ", ""), metric)
                name = name.lower().replace(" ", "") if name != "test" else dataset_name.lower().replace(" ", "")
            
                # Pair of label & value
                labels.append( name.title() )
                values.append( float(results[key]) )

            # Color palette
            colors = list(sns.color_palette(cc.glasbey, n_colors = len(labels)))

            plt.sca(axes.flat[1])
            bars = plt.bar( labels, values, color = colors, width = 0.4 )
            plt.bar_label(bars)
            plt.title("{} per Dataset".format(metric.title()), fontsize = 24)
            plt.xlabel("Dataset", fontsize = 20)
            plt.xticks(fontsize = 16, rotation = 45)
            plt.ylabel(metric.title(), fontsize = 20)
            plt.ylim( (np.min( [0.7, .95*np.min( values )] ), 1.0) )
            fig.savefig( plot_path, bbox_inches = "tight" )
            plt.close(fig)

        return

    @staticmethod
    def dict_hash( src_dict ) :
        """ MD5 hash of a dictionary.
        Based on: https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
        """
        dhash = hashlib.md5()
        encoded = json.dumps(src_dict, sort_keys=True).encode()
        dhash.update(encoded)
        return dhash.hexdigest()