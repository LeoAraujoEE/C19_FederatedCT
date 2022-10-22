import os
import sys
import time
import json
import random
import numpy as np
import tensorflow as tf

# Enabled deterministic mode/disables multiprocessing to enforce determinism
os.environ["PYTHONHASHSEED"] = str(0)
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
from utils.custom_model_trainer import ModelTrainer

# Decodes all the input args and creates a dict
arg_dict = json.loads(sys.argv[1])

# Extract info from from args_dict
seed               = arg_dict.pop("seed")
verbose            = arg_dict.pop("verbose")
train_dataset      = arg_dict.pop("dataset")
import_dir         = arg_dict.pop("data_path")
dst_dir            = arg_dict.pop("output_dir")
model_id           = arg_dict.pop("model_hash")
model_fname        = arg_dict.pop("model_filename")
ignore_check       = arg_dict.pop("ignore_check")
keep_pneumonia     = arg_dict.pop("keep_pneumonia")
hyperparameters    = arg_dict.pop("hyperparameters")
data_aug_params    = arg_dict.pop("data_augmentation")
save_final_weights = arg_dict.pop("save_final_weights")
remove_unfinished  = arg_dict.pop("remove_unfinished")
current_epoch_num  = arg_dict.pop("current_epoch_num")
epochs_per_step    = arg_dict.pop("epochs_per_step")
max_train_steps    = arg_dict.pop("max_train_steps")
initial_weights    = arg_dict.pop("initial_weights")

# Setting seeds to enforce deterministic behaviour
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
tf.experimental.numpy.random.seed(seed)
        
# Builds object to handle the training dataset
dataTrain = Dataset( import_dir, train_dataset, keep_pneumonia )

# Initializes trainer object
trainer = ModelTrainer(dst_dir, dataTrain, model_fname, 
                       model_id, save_final_weights)

if trainer.check_step( ignore_check ):
  
  # Removes folders of models whose testing process did not finish
  # Ignored during Federated Learning as local models aren't tested
  trainer.prepare_model_dir(remove_unfinished)

  # Prints current hyperparameters
  if verbose:
    print(f"\nUsing dataset: '{trainer.dataset.name}' and:")
    trainer.print_dict( hyperparameters, round = True )
  
  # Starts training
  history, train_time = trainer.train_model(hyperparameters, data_aug_params,
                                            initial_epoch = current_epoch_num, 
                                            epochs_per_step = epochs_per_step, 
                                            # max_steps = max_train_steps,
                                            max_steps = 3,
                                            load_from = initial_weights)

  # Saves history_dict as CSV
  trainer.history_to_csv(history)

  #
  print("\nSaving training hyperparameters as JSON...")
  trainer.hyperparam_to_json(hyperparameters, data_aug_params, train_time)