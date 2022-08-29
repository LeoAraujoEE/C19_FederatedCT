import os
from utils.custom_model_trainer import ModelManager

PATH_DICT = { "datasets": os.path.join( "D:\\", "Datasets", "COVID19", "CT", "classification" ),
              "outputs" : os.path.join( ".", "output", "models" ) 
            }

train_dataset = "COVID-CT-MD"
model_fname = "efficientnetv2_b0_94de7f456996e39af676173246d61888"

json_path = os.path.join( ".", "output", "models", "radiopaedia_dropped", 
                          model_fname, f"params_{model_fname}.json" )

  
trainManager = ModelManager( path_dict = PATH_DICT, 
                             dataset_name = train_dataset, 
                             keep_pneumonia = False )

trainManager.doTrainFromJSON( json_path, copy_augmentation = True )
