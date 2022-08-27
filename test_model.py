import os
import sys
import tensorflow as tf

# Allows memory growth for GPU usage
for device in tf.config.list_physical_devices("GPU"):
  try:
    tf.config.experimental.set_memory_growth( device, True )
  except:
    pass

# Imports from other scripts
from utils.dataset import load_datasets
from utils.custom_plots import CustomPlots
from utils.custom_model_trainer import ModelTrainer

# Decodes all the input args and creates a dict
arg_dict = ModelTrainer.decode_args(sys.argv)

# Setting seeds to enforce deterministic behaviour in hashing processes
os.environ["PYTHONHASHSEED"] = str(0)

# Builds object to handle datasets for training and for external validation
dataTrain, dataVal_list = load_datasets( import_dir = arg_dict["data_path"], 
                                         train_dataset = arg_dict["train_dataset"], 
                                         input_col = "path", output_col = "class", 
                                         keep_pneumonia = arg_dict["keep_pneumonia"] )

trainer = ModelTrainer( dataTrain, dataVal_list, dst_dir = arg_dict["output_dir"] )

# Extract model's hash and model's filename from args_dict
model_id = arg_dict["model_hash"]
model_fname = arg_dict["model_filename"]

# Extracts hyperparameters and parameters for data augmentation from args_dict
hyperparameters, data_aug_params = trainer.get_dicts(arg_dict)

if trainer.check_step( model_id, ignore = arg_dict["ignore_check"] ):
  
  # Removes models whose training process did not finish properly
  trainer.remove_unfinished()

  # Generates model path
  model_path, model_fname = trainer.get_model_path( model_fname, model_id )

  # Object responsible for plotting
  trainer.plotter = CustomPlots(model_path)

  # Gets the names of the datasets used in training/testing the models
  dataset_name = trainer.dataset.name
  cval_dataset_names = [ dset.name for dset in trainer.dataset_list ]

  # Prints current hyperparameters
  trainer.print_dict( hyperparameters, round = True )

  # Announces the start of the testing process
  print(f"\nTesting model '{os.path.basename(model_path)}'...")
  results_dict = trainer.test_model( model_path, hyperparameters )

  print("\nPlotting test results...")
  trainer.plotter.plot_test_results( results_dict, dataset_name, cval_dataset_names )

  # Prints the results
  print("\nTest Results:")
  trainer.print_dict( results_dict, round = True )

  # Saves the results to a CSV file
  print("\nSaving model hyperparameters and results as CSV...")
  trainer.append_to_csv( model_path, model_id, hyperparameters, data_aug_params, results_dict )