import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
for device in tf.config.list_physical_devices("GPU"):
  try:
    tf.config.experimental.set_memory_growth( device, True )
  except:
    pass

from utils.dataset import load_datasets
from utils.custom_model_trainer import ModelTrainer

# Decodes all the input args and creates a dict
arg_dict = ModelTrainer.decode_args(sys.argv)

# Builds object to handle datasets for training and for external validation
dataTrain, dataVal_list = load_datasets( import_dir = arg_dict["data_path"], 
                                         train_dataset = arg_dict["train_dataset"], 
                                         input_col = "path", output_col = "class", 
                                         keep_pneumonia = arg_dict["keep_pneumonia"] )

trainer = ModelTrainer( dataTrain, dataVal_list, dst_dir = arg_dict["output_dir"] )
trainer.train_test_iteration( arg_dict )
