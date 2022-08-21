import os
from utils.custom_model_trainer import ModelManager

PATH_DICT = { "datasets": os.path.join( "D:\\", "Datasets", "COVID19", "CT", "classification" ),
              "outputs" : os.path.join( "." ) }

train_dataset = "radiopaedia.org"
reference_dataset = "Comp_CNCB_iCTCF_a_dropped"
reference_metrics = ["test_f1"]

trainManager = ModelManager( path_dict = PATH_DICT, 
                             dataset_name = train_dataset, 
                             keep_pneumonia = False )

trainManager.doJsonSearch( reference_dataset, reference_metrics )
