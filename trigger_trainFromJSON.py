import os
from utils.custom_model_trainer import ModelManager

PATH_DICT = { "datasets": os.path.join( "C:\\", "Datasets", "COVID19", "CT", "classification" ),
              "outputs" : os.path.join( ".", "output", "federated_models" ) 
            }

train_dataset = "miniCOVIDxCT"
model_fname = "fl_resnet18_6b11d9834590cb2dcbc8686d0ac496a3"

json_path = os.path.join( ".", "output", "federated_models", #"models", "COVID-CTset", 
                          model_fname, f"params_{model_fname}.json" )

  
trainManager = ModelManager( path_dict = PATH_DICT, 
                             dataset_name = train_dataset, 
                             keep_pneumonia = False )

trainManager.doTrainFromJSON( json_path, copy_augmentation = True )
