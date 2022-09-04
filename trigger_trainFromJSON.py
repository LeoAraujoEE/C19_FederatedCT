import os
from utils.custom_model_trainer import ModelManager

PATH_DICT = { "datasets": os.path.join( "C:\\", "Datasets", "COVID19", "CT", "classification" ),
              "outputs" : os.path.join( ".", "output", "deterministic", "models" ) 
            }

train_dataset = "COVID-CT-MD"
model_fname = "efficientnetv2_b0_d9ee118fbab542200f090c0da1429430"

json_path = os.path.join( ".", "output", "deterministic", "models", 
                          model_fname, f"params_{model_fname}.json" )

  
trainManager = ModelManager( path_dict = PATH_DICT, 
                             dataset_name = train_dataset, 
                             keep_pneumonia = False )

for i in range(2):
    trainManager.doTrainFromJSON( json_path, copy_augmentation = True )
