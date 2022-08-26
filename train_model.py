import os
import sys
import time
import random
import warnings
import numpy as np
import tensorflow as tf

# Disables warning messages
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Enabled deterministic mode/disables multiprocessing to enforce determinism
os.environ['TF_CUDNN_DETERMINISTIC'] = 'True'
tf.config.experimental.enable_op_determinism()
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

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

# Setting seeds to enforce deterministic behaviour
random.seed(arg_dict["seed"])
np.random.seed(arg_dict["seed"])
tf.random.set_seed(arg_dict["seed"])
tf.keras.utils.set_random_seed(arg_dict["seed"])
tf.experimental.numpy.random.seed(arg_dict["seed"])
os.environ["PYTHONHASHSEED"] = str(0)

# Builds object to handle datasets for training and for external validation
dataTrain, dataVal_list = load_datasets( import_dir = arg_dict["data_path"], 
                                         train_dataset = arg_dict["train_dataset"], 
                                         input_col = "path", output_col = "class", 
                                         keep_pneumonia = arg_dict["keep_pneumonia"] )

trainer = ModelTrainer( dataTrain, dataVal_list, dst_dir = arg_dict["output_dir"] )

# Extracts hyperparameters and parameters for data augmentation from args_dict
hyperparameters, data_aug_params = trainer.get_dicts(arg_dict)

if trainer.check_step( hyperparameters, data_aug_params, 
                       ignore = arg_dict["ignore_check"] ):
  
  # Removes models whose training process did not finish properly
  trainer.remove_unfinished()

  # Generates model path and the model id
  model_path, model_id = trainer.gen_model_name( hyperparameters, data_aug_params )

  # Object responsible for plotting
  trainer.plotter = CustomPlots(model_path)

  # Prints current hyperparameters and starts training
  trainer.print_dict( hyperparameters, round = True )
  train_start_t = time.time()
  history_dict  = trainer.train_model( hyperparameters, data_aug_params, model_path )

  # Records the total training time
  ellapsed_time = (time.time() - train_start_t)
  train_time = trainer.ellapsed_time_as_str(ellapsed_time)

  # Saves history_dict to CSV
  trainer.history_to_csv(history_dict, model_path)
          
  # Gets the names of the datasets used in training/testing the models
  dataset_name = trainer.dataset.name
  cval_dataset_names = [ dset.name for dset in trainer.dataset_list ]

  # Announces the end of the training process
  print(f"\nTrained model '{os.path.basename(model_path)}' in {train_time}. Plotting results...")
  trainer.plotter.plot_train_results( history_dict, dataset_name )

  # Prints current hyperparameters
  trainer.print_dict( hyperparameters, round = True )

  # Announces the start of the testing process
  print(f"\nTesting model '{os.path.basename(model_path)}'...")
  results_dict = trainer.test_model( model_path, hyperparameters )
  results_dict["train_time"] = train_time

  print("\nPlotting test results...")
  trainer.plotter.plot_test_results( results_dict, dataset_name, cval_dataset_names )

  # Prints the results
  print("\nTest Results:")
  trainer.print_dict( results_dict, round = True )

  # Saves the results to a CSV file
  print("\nSaving model hyperparameters and results as CSV...")
  trainer.append_to_csv( model_path, model_id, hyperparameters, data_aug_params, results_dict )

  #
  print("\nSaving training hyperparameters as JSON...")
  trainer.hyperparam_to_json(model_path, hyperparameters, data_aug_params)

else:
    print("\tStep already executed: Skipping...")