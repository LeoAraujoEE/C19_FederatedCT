import os
from utils.custom_model_trainer import ModelManager

KEEP_PNEUMONIA = True
SUBDIR = "Remapped" if KEEP_PNEUMONIA else "Dropped"

# 
PATH_DICT = { "datasets": os.path.join( "C:\\", "Datasets", "COVID19", "CT", "classification" ),
              "outputs" : os.path.join( ".", "output", SUBDIR, "mock" ),
              # "outputs" : os.path.join( ".", "output", SUBDIR, "models" ) 
            }

train_dataset = "COVID-CT-MD"
model_fname = "resnet18_c6f4e288062b353ba4ae5a6d44ef5919"

json_path = os.path.join( PATH_DICT["outputs"], f"COVID-CT-MD_{SUBDIR.lower()}", 
                          model_fname, f"params_{model_fname}.json" )
  
trainManager = ModelManager( path_dict = PATH_DICT, 
                             dataset_name = train_dataset, 
                             keep_pneumonia = KEEP_PNEUMONIA )

trainManager.doTrainFromJSON( json_path, copy_augmentation = True )
