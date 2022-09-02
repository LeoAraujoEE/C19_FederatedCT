import os
import sys
import time
import random
import numpy as np
import tensorflow as tf

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
from utils.dataset import Dataset
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
        
# Builds object to handle the training dataset
dataTrain = Dataset( arg_dict["data_path"], name = arg_dict["dataset"], 
                     keep_pneumonia = arg_dict["keep_pneumonia"] )

trainer = ModelTrainer( dataTrain, dst_dir = arg_dict["output_dir"] )

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

  # Prints current hyperparameters
  trainer.print_dict( hyperparameters, round = True )
  
  # Starts training
  history_dict  = trainer.train_model( hyperparameters, data_aug_params, model_path, 
                                       initial_epoch = arg_dict["current_epoch_num"], 
                                       epochs_per_step = arg_dict["epochs_per_step"], 
                                       max_steps = arg_dict["max_train_steps"],
                                       load_from = arg_dict["load_from"] )

  # Saves history_dict to CSV
  trainer.history_to_csv(history_dict, model_path)

  # Announces the end of the training process
  print(f"\nTrained model '{model_fname}'. Plotting train results...")
  trainer.plotter.plot_train_results( history_dict, trainer.dataset.name )

  #
  print("\nSaving training hyperparameters as JSON...")
  trainer.hyperparam_to_json(model_path, hyperparameters, data_aug_params)