import os
import sys
import json
import tensorflow as tf

# Allows memory growth for GPU usage
for device in tf.config.list_physical_devices("GPU"):
  try:
    tf.config.experimental.set_memory_growth( device, True )
  except:
    pass

# Imports from other scripts
from utils.dataset import load_datasets
from utils.custom_model_trainer import ModelTester

# Decodes all the input args and creates a dict
arg_dict = json.loads(sys.argv[1])

# Extract info from from args_dict
seed                = arg_dict.pop("seed")
verbose             = arg_dict.pop("verbose")
train_dataset       = arg_dict.pop("dataset")
dataset_dir         = arg_dict.pop("data_path")
dst_dir             = arg_dict.pop("output_dir")
model_id            = arg_dict.pop("model_hash")
model_fname         = arg_dict.pop("model_filename")
ignore_check        = arg_dict.pop("ignore_check")
keep_pneumonia      = arg_dict.pop("keep_pneumonia")
hyperparameters     = arg_dict.pop("hyperparameters")
data_aug_params     = arg_dict.pop("data_augmentation")
use_validation_data = arg_dict.pop("use_validation_data")

# Setting seeds to enforce deterministic behaviour in hashing processes
os.environ["PYTHONHASHSEED"] = str(seed)

# Builds object to handle datasets for training and for external validation
dataTrain, dataVal_list = load_datasets( dataset_dir, train_dataset, 
                                         use_validation_data, keep_pneumonia )

tester = ModelTester( dst_dir, model_fname, model_id, dataTrain, 
                      dataVal_list, use_validation_data )

if tester.check_step( ignore_check ):

  # Prints current hyperparameters
  if verbose:
    print(f"\nTesting model '{tester.model_fname}':")
    tester.print_dict( hyperparameters, round = True )

  # Starts the testing process
  results_dict = tester.test_model(hyperparameters)

  # Prints the results
  if not use_validation_data:
    print(f"\nTest Results:")
    tester.print_dict( results_dict, round = True )

  # Saves the results to a CSV file
  if verbose:
    print("\nSaving model hyperparameters and results as CSV...")
  tester.append_to_csv(hyperparameters, data_aug_params, results_dict)