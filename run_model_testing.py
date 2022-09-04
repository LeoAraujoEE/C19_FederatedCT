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

# Setting seeds to enforce deterministic behaviour in hashing processes
os.environ["PYTHONHASHSEED"] = str(0)

# Builds object to handle datasets for training and for external validation
dataTrain, dataVal_list = load_datasets( import_dir = arg_dict["data_path"], 
                                         train_dataset = arg_dict["dataset"], 
                                         eval_partition = arg_dict["eval_partition"], 
                                         keep_pneumonia = arg_dict["keep_pneumonia"] )

tester = ModelTester( dst_dir = arg_dict["output_dir"], dataset = dataTrain, 
                      dataset_list = dataVal_list )

# Extract model's hash and model's filename from args_dict
model_id = arg_dict["model_hash"]
model_fname = arg_dict["model_filename"]

# Extracts hyperparameters and parameters for data augmentation from args_dict
hyperparameters = arg_dict["hyperparameters"]
data_aug_params = arg_dict["data_augmentation"]

if tester.check_step( model_id, ignore = arg_dict["ignore_check"] ):

  # Generates model path
  model_path, model_fname = tester.get_model_path( model_fname, model_id )

  # Gets the names of the datasets used in training/testing the models
  dataset_name = tester.dataset.name
  cval_dataset_names = [ dset.name for dset in tester.dataset_list ]

  # Prints current hyperparameters
  print(f"\nTesting model '{os.path.basename(model_path)}':")
  tester.print_dict( hyperparameters, round = True )

  # Starts the testing process
  results_dict = tester.test_model(model_path, hyperparameters,
                         eval_part = arg_dict["eval_partition"])

  # Prints the results
  print("\nTest Results:")
  tester.print_dict( results_dict, round = True )

  # Saves the results to a CSV file
  print("\nSaving model hyperparameters and results as CSV...")
  tester.append_to_csv(model_path, model_id, hyperparameters, 
                       data_aug_params, results_dict)