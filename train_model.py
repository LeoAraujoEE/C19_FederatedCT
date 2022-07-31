import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

from utils.dataset import load_datasets
from utils.custom_model_trainer import ModelTrainer

# Decodes all the input args and creates a dict
arg_dict = ModelTrainer.decode_args(sys.argv)

# Path to resized datasets in COVIDx CT-3A
path2datasets = arg_dict["data_path"]

# Checks the desired input shape and uses resized images if Height and Width are set to 256
input_column = "path_256" if (arg_dict["input_height"] == 256) and (arg_dict["input_width"] == 256) else "path_512"

# Builds object to handle datasets for training and for external validation
dataTrain, dataVal_list = load_datasets( import_dir = path2datasets, train_dataset = arg_dict["train_dataset"], 
                                         input_col = input_column, output_col = "class", 
                                         keep_pneumonia = arg_dict["keep_pneumonia"] )

trainer = ModelTrainer( dataTrain, dataVal_list, dst_dir = arg_dict["output_dir"] )
trainer.train_test_iteration( arg_dict )
