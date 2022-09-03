import os

# Imports from other scripts
from utils.dataset import Dataset

# 
path_dict = { "datasets": os.path.join( "C:\\", "Datasets", "COVID19", "CT", "classification" ),
              "outputs" : os.path.join( ".", "output", "models" ) 
            }
        
# Builds object to handle the training dataset
dataTrain = Dataset( path_dict["datasets"], name = "miniCOVIDxCT", 
                     keep_pneumonia = False )

dataTrain.load_dataframes()

print( f"\n{dataTrain.name}:" )
for part in ["train", "val", "test"]:
    df = dataTrain.get_dataframe(part)
    base_path = os.path.join(path_dict["datasets"], "224x224")
    paths = [os.path.join(base_path, p) for p in df["path"].to_list()]
    paths = [p for p in paths if os.path.exists(p)]
    
    print("\n")
    print( f"\t{part.title()}: {dataTrain.get_num_samples(part)} (JSON)" )
    print( f"\t{part.title()}: {len(df)} (len(df))" )
    print( f"\t{part.title()}: {len(df)} (existant paths)" )
    