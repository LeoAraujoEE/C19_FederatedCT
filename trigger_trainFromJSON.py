import os
from utils.custom_model_trainer import ModelManager

KEEP_PNEUMONIA = True
SUBDIR = "remapped" if KEEP_PNEUMONIA else "dropped"

# 
PATH_DICT = { "datasets": os.path.join( "..", "data", "Processed", "CT", "classification", "COVIDxCT-3A" ),
              "outputs" : os.path.join( "..", "output", "CT", "mock_models", SUBDIR ), 
              # "outputs" : os.path.join( "..", "output", "CT", "models", SUBDIR ) 
            }

train_dataset = "COVID-CT-MD"
model_fname = "resnet18_c6f4e288062b353ba4ae5a6d44ef5919"

json_path = os.path.join( PATH_DICT["outputs"], f"COVID-CT-MD_{SUBDIR.lower()}", 
                          model_fname, f"params_{model_fname}.json" )
  
trainManager = ModelManager( path_dict = PATH_DICT, 
                             dataset_name = train_dataset, 
                             keep_pneumonia = KEEP_PNEUMONIA )

for i in range(2):
    trainManager.doTrainFromJSON( json_path, copy_augmentation = True )
