import os

# Imports from other scripts
from utils.dataset import Dataset

# 
path_dict = { "datasets": os.path.join( "D:\\", "Datasets", "COVID19", "CT", "classification" ),
              "outputs" : os.path.join( ".", "output", "models" ) 
            }
        
# Builds object to handle the training dataset
dataTrain = Dataset( path_dict["datasets"], name = "radiopaedia.org", 
                     keep_pneumonia = False )

print( dataTrain.name )