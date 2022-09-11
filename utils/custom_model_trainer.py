import os
import warnings

# Suppresses warning messages
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import json
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
        
        # Checks wether hyperparameter values were provided
        if isinstance(hyperparam_values, dict):
            # Prints possible values for training hyperparameters
            print("\nList of possible hyperparameter values:")
            self.print_dict(hyperparam_values)
            
            # If running a Federated Learning Simulation
            if self.federated:
                # Combines fl_params and hyperparam_values into a single dict
                fl_params.update(hyperparam_values)
                self.hyperparam_values = fl_params
                
            else:
                # Else, uses hyperparam_values as training hyperparameters
                self.hyperparam_values = hyperparam_values
        
        # Wether to keep pneumonia sample or remove them
        self.keep_pneumonia = keep_pneumonia
        
        # Directory for all available datasets
        self.data_path = path_dict["datasets"]
        
        # Train dataset for regular training
        # Validation dataset for Federated Learning simulations
        self.dataset = Dataset( self.data_path, name = dataset_name, 
                                keep_pneumonia = self.keep_pneumonia )
        
        # Directory for the output trained models
        if self.federated:
            self.dst_dir = path_dict["outputs"]
        # If FL training isn't being simulated, a new subdir is added to split
        # the resulting models based on which dataset they're trained with
        else:
            self.dst_dir = os.path.join( path_dict["outputs"], 
                                         self.dataset.name )
        
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
    
    def check_trainability(self, list_values = True):
        assert not (self.hyperparam_values is None), "\nHyperparameter values were not provided..."
        assert not (self.aug_params is None), "\nData augmentation parameters were not provided..."
        if list_values:
            txt = "All hyperparameter values should be lists..."
            assert all([isinstance(v, list) for v in self.hyperparam_values.values()]), txt
        else:
            txt = "All hyperparameter values should be tuples..."
            assert all([isinstance(v, tuple) for v in self.hyperparam_values.values()]), txt
        return True

    def doGridSearch( self, shuffle = False ):
        self.check_trainability(list_values = True)
        print("\nStarting GridSearch:")

        # Checks all possible permutations from the given values
        hyperparam_permutations = list(self.product_dict(**self.hyperparam_values))
        print(f"\nA total of {len(hyperparam_permutations)} hyperparameter permutations were found.")
        
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
        self.check_trainability(list_values = False)
        print("\nStarting RandomSearch:")

        idx_h = 0
        while idx_h < n_models:
            hyperparameters = self.gen_random_hyperparameters(self.hyperparam_values)

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
        hyperparameters, aug_params = self.json_to_hyperparam( json_path )

        # Copies the augmentation dict used if specified
        if copy_augmentation:
            self.aug_params = aug_params
        
        # Changes the random seed used if specified
        if not seed is None:
            hyperparameters["seed"] = seed
            
        if self.federated:
            # Trains model simulating Federated Learning
            self.run_process( self.fl_code, hyperparameters, 
                              ignore_check = True )
        
        else:
            # Trains model
            self.run_process( self.train_code, hyperparameters, 
                              ignore_check = True )
            
            # Tests model
            self.run_process( self.test_code, hyperparameters, 
                              ignore_check = True )

        return
            
    def run_process(self, script_code, hyperparams, ignore_check):
        assert script_code in [0, 1, 2], f"Unknown script w/ code {script_code}..."
        
        # Creates test command
        command = self.create_command(hyperparams, ignore_check, script_code)

        # Runs script as subprocess
        subprocess.Popen.wait(subprocess.Popen( command ))
        return

    def create_command(self, hyperparams, ignore_check, script_code):
        
        model_fname, model_id = self.get_model_name( hyperparams, 
                                                     self.aug_params )
        
        # Base args dict
        args = { "output_dir"       :           self.dst_dir, 
                 "data_path"        :         self.data_path,
                 "dataset"          : self.dataset.orig_name, 
                 "keep_pneumonia"   :    self.keep_pneumonia,
                 "ignore_check"     :           ignore_check,
                 "model_hash"       :               model_id, 
                 "model_filename"   :            model_fname,
                 "hyperparameters"  :            hyperparams,
                 "data_augmentation":        self.aug_params,
                 "verbose"          :                      1,
                 "seed"             :    hyperparams["seed"],
               }
        
        # Matches the recieved code with one of the flags
        if script_code == self.train_code:
            # Sets the command for training process
            script = "run_model_training"
            
            # Updates args dict w/ training arguments
            args["initial_weights"]   = None
            args["max_train_steps"]   = None
            args["epochs_per_step"]   = None
            args["current_epoch_num"] =    0
        
        elif script_code == self.test_code:
            # Sets the command for testing process
            script = "run_model_testing"
            
            # Updates args dict w/ testing arguments
            args["eval_partition"] = "test"
        
        elif script_code == self.fl_code:
            # Sets the command for Federated Learning simulation
            script = "run_fl_simulation"
        
        # Serializes args dict as JSON formatted string
        serialized_args = json.dumps(args)
        
        # 
        command = ["python", "-m", script, serialized_args]

        return command

    def get_model_name( self, hyperparameters, aug_params ):
        # Creates a single dict to store all hyperparameters
        all_param_dict = {}

        # Iterates through existing dicts
        for param_dict in [ hyperparameters, aug_params]:
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
            model_fname = "fl_" + model_fname

        return model_fname, model_id
    
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
        hyperparameters = data["hyperparameters"]
        augmentation_params = data["augmentation_params"]

        return hyperparameters, augmentation_params

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
        self.model_path, self.model_fname = self.get_model_path( model_fname,
                                                                 model_id )
        
        # Sets model_id and model_dir
        self.model_id = model_id
        self.model_dir = os.path.join( self.dst_dir, self.model_fname )
            
        # Initializes other class variables
        self.model = None
            
        return

    def prepare_model_dir(self):
        print("\nPreparing model dir:")

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
            weights_path = os.path.join(path2subdir, f"{model_basename}.h5")

            if (weights_path in finished_models):
                continue

            print(f"\tDeleting '{model_basename}' subdir as its training did not finish properly...")
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

    def get_model_path( self, model_fname, model_id ):

        # Creates the full model path
        model_path = os.path.join( self.dst_dir, model_fname, 
                                   f"{model_fname}.h5" )
        
        # Checks if a model with the same name already exists
        # possible if a combination of hyperparameters is being retrained
        if os.path.exists(os.path.dirname(model_path)):
            
            idx = 0
            if os.path.exists(self.csv_path):
                # Reads CSV file to count models w/ same hash
                result_df = pd.read_csv(self.csv_path, sep = ";")

                # Counts the amount of models with the same hash
                idx = len(result_df[result_df["model_hash"] == model_id])
            
            # Keeps the same path / fname if there're no entries
            if idx > 0:
                # Otherwise, updates model_fname and model_path 
                # to avoid overwritting the existant model
                model_fname = f"{model_fname}_{idx+1}"
                model_path = os.path.join(self.dst_dir, model_fname,
                                          f"{model_fname}.h5")

        return model_path, model_fname

    def prepare_model( self, hyperparameters ):
        assert not self.model is None, "There's no model to prepare..."

        # Compiles the model
        f1_metric = F1Score( num_classes = 1, threshold = .5, average = "micro", name = "f1" )
        if hyperparameters["optimizer"].lower() == "adam":
            self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hyperparameters["start_lr"]), 
                               loss = "binary_crossentropy", metrics = ["acc", f1_metric])
        else:
            self.model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = hyperparameters["start_lr"]), 
                               loss = "binary_crossentropy", metrics = ["acc", f1_metric])
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

class ModelTrainer(ModelHandler):
    def __init__(self, dst_dir, dataset, model_fname, model_id):
        
        # Inherits ModelHandler's init method
        ModelHandler.__init__( self, dst_dir, model_fname, model_id )

        # Sets the dataset used for training
        self.dataset = dataset
        self.dataset.load_dataframes()

        return

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

        # Callbacks --------------------------------------------------------------------------------------------------
        # List of used callbacks
        callback_list = [] 

        # Adds Model Checkpoint/Early Stopping if a monitor variable is passed
        var_list = ["val_loss", "val_acc", "val_f1"]
        if hyperparameters["monitor"] in var_list:
            
            # Sets callback_mode based on the selected monitored metric
            callback_mode = "min" if "loss" in hyperparameters["monitor"].lower() else "max"
            
            # Model Checkpoint
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_path,
                                      monitor = hyperparameters["monitor"],
                                      mode = callback_mode, 
                                      save_best_only = True,
                                      save_weights_only = True, 
                                      include_optimizer = False, verbose = 1)
            callback_list.append(model_checkpoint)
            
            # Early Stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                                    monitor = hyperparameters["monitor"],
                                    patience = hyperparameters["early_stop"],
                                    mode = callback_mode, verbose = 1 )
            callback_list.append(early_stopping)
        
        # Learning Rate Scheduler
        def scheduler(epoch, lr):
                
            # Number of completed steps
            steps = (epoch + 1) // hyperparameters["lr_adjust_freq"]
            
            # Coeficient to multiply initial lr and get the new lr
            coef = hyperparameters["lr_adjust_frac"] ** steps
            
            # Gets the new lr value and prints the change
            new_lr = hyperparameters["start_lr"] * coef
            
            # Prints only in the epochs where the lr is changed
            if (epoch + 1) % hyperparameters["lr_adjust_freq"] == 0:
                print(f"[LR Scheduler] Updating LearningRate from '{lr:.3E}' to '{new_lr:.3E}'...")
                
            return new_lr
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, 
                                                                verbose = 0)
        callback_list.append(lr_scheduler)
        # Callbacks --------------------------------------------------------------------------------------------------

        # Creates train data generator
        train_datagen = CustomDataGenerator( self.dataset, "train", hyperparameters, aug_dict = aug_params, shuffle = True, 
                                             undersample = hyperparameters["apply_undersampling"], seed = hyperparameters["seed"] )

        # Creates validation data generator
        val_datagen   = CustomDataGenerator( self.dataset, "val", hyperparameters, undersample = False, shuffle = False )

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
            # For Federated Learning simulations, each step performs only a 
            # few epochs. 
            num_epochs = initial_epoch + epochs_per_step

        # Fits the model
        history = self.model.fit( x = train_datagen, steps_per_epoch = train_steps, 
                                  epochs = num_epochs, initial_epoch = initial_epoch, 
                                  validation_data = val_datagen, validation_steps = val_steps, 
                                  callbacks = callback_list, class_weight = class_weights, 
                                  verbose = 1
                                )

        # Saves model configs
        json_config = self.model.to_json()
        config_path = self.model_path.replace(".h5", ".json")

        with open(config_path, "w") as json_file:
            json.dump( json_config, json_file, indent=4 )
        
        # Saves weights if model_checkpoint is disabled
        if not hyperparameters["monitor"] in var_list:
            self.model.save_weights( self.model_path )

        # Extracts the dict with the training and validation values for loss and IoU during training
        history_dict = history.history
  
        # Object responsible for plotting
        print(f"\nTrained model '{self.model_fname}'...")
        plotter = CustomPlots(self.model_path)
        plotter.plot_train_results( history_dict, self.dataset.name )

        return history_dict
    
    def history_to_csv(self, history_dict):

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
    
    def hyperparam_to_json( self, hyperparameters, aug_params ):

        # Builds a dict of dicts w/ hyperparameters needed to reproduce a model
        dict_of_dicts = { "hyperparameters"    : hyperparameters, 
                          "augmentation_params": aug_params,
                        }
        
        json_name = f"params_{self.model_fname}.json"
        json_path = os.path.join( self.model_dir, json_name )

        # Saves the JSON file
        with open(json_path, "w") as json_file:
            json.dump( dict_of_dicts, json_file, indent=4 )

        return
    
    @staticmethod
    def ellapsed_time_as_str( seconds ):
        int_secs  = int(seconds)
        str_hours = str(int_secs // 3600).zfill(2)
        str_mins  = str((int_secs % 3600) // 60).zfill(2)
        str_secs  = str(int_secs % 60).zfill(2)
        time_str  = f"{str_hours}:{str_mins}:{str_secs}"
        return time_str

class ModelTester(ModelHandler):
    def __init__(self, dst_dir, model_fname, model_id, 
                 dataset = None, dataset_list = None):
        
        # Inherits ModelHandler's init method
        ModelHandler.__init__( self, dst_dir, model_fname, model_id )

        # Sets the dataset used for training if available
        self.dataset = dataset
        if not self.dataset is None:
            self.dataset.load_dataframes()

        # Sets the datasets used for cross-validation if available
        self.dataset_list = dataset_list
        if not self.dataset_list is None:
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
        num_steps = dataset.get_num_steps(partition, hyperparameters["batchsize"])

        # Creates data generator and gets all the labels as an array
        datagen = CustomDataGenerator( dataset, partition, hyperparameters, shuffle = False, undersample = False )

        # Gets all labels in the dataframe as their corresponding class numbers to compute accuracy and f1-score
        y_true = datagen.get_labels()[:num_samples]

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

    def test_model( self, hyperparameters, eval_part = "test" ):

        # Object responsible for plotting ROC curves and Confusion Matrixes
        plotter = CustomPlots(self.model_path)

        # Loads model
        print(f"\nLoading model from '{self.model_path}'...")
        self.model = self.load_model( self.model_path )
        self.prepare_model( hyperparameters )

        # Creates a dictionary with ordered keys, but no values
        results = self.get_base_results_dict()

        # Evaluates each partition to fill results dict
        if not self.dataset is None:
            # Announces the dataset used for training
            dataset_name = self.dataset.name
            print(f"\nValidating model '{self.model_fname}' on '{dataset_name}' dataset...")
        
            for partition in ["train", "val", "test"]:
                print(f"\n\n{partition.title()}:")
                acc, f1, auroc, conf_matrix, y_true, y_preds = self.evaluate_model( self.dataset, hyperparameters, partition )

                for metric, value in zip( ["acc", "f1", "auc"], [acc, f1, auroc] ):
                    # Adds the results to the result dict
                    key = f"{partition}_{metric}"
                    results[key] = f"{value:.4f}"

                # Plots confusion matrix
                class_labels = self.dataset.classes
                plotter.plot_confusion_matrix(conf_matrix, dataset_name, 
                                            partition, class_labels)

                # Plots ROC curves
                plotter.plot_roc_curve(y_true, y_preds, dataset_name, partition)

        # If there are datasets for cross-validation
        cval_dataset_names = []
        if not self.dataset_list is None:
            cval_acc_list, cval_f1_list, cval_auroc_list = [], [], []

            for dset in self.dataset_list:
                dset_name = dset.name
                cval_dataset_names.append(dset_name)
                
                # Announces the dataset used for testing
                print(f"\nCross-Validating model '{self.model_fname}' on '{dset_name}' dataset ({eval_part})...")

                # Evaluates dataset
                acc, f1, auroc, conf_matrix, y_true, y_preds = self.evaluate_model( dset, hyperparameters, 
                                                                                    eval_part )

                # Adds to list
                cval_acc_list.append(acc)
                cval_f1_list.append(f1)
                cval_auroc_list.append(auroc)

                for metric, value in zip( ["acc", "f1", "auc"], [acc, f1, auroc] ):
                    # Adds the results to the result dict
                    dname = dset_name.lower().replace(" ", "")
                    key = f"{dname}_{metric}"
                    results[key] = f"{value:.4f}"
                    
                if eval_part == "test":
                    # Plots confusion matrix
                    class_labels = dset.classes
                    plotter.plot_confusion_matrix(conf_matrix, dset_name, 
                                                  eval_part, class_labels)

                    # Plots ROC curves
                    plotter.plot_roc_curve(y_true, y_preds, dset_name, eval_part)
                
            results["crossval_acc"] = f"{np.mean(cval_acc_list):.4f}"
            results["crossval_f1"] = f"{np.mean(cval_f1_list):.4f}"
            results["crossval_auc"] = f"{np.mean(cval_auroc_list):.4f}"

        if (not self.dataset is None) and (eval_part == "test"):
            plotter.plot_test_results(results, dataset_name, cval_dataset_names)

        return results
    
    def get_base_results_dict( self ):
        # Generates keys and instantiates their value as None
        results = {}
        for metric in ["acc", "f1", "auc"]:
            # If there's a train dataset to evaluate
            if not self.dataset is None:
                # Generates 1 entry for each metric for each partition
                for partition in ["train", "val", "test"]:
                    key = f"{partition}_{metric}"
                    results[key] = None

            # If there are datasets for cross-validation
            if not self.dataset_list is None:
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
        combined_dict = { "model_path": self.model_path, 
                          "model_hash": self.model_id 
                        }
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
    