import os
import json
import glob
import numpy as np
import pandas as pd

class Dataset():
    def __init__( self, import_dir, name, keep_pneumonia = False ):
        """ Initializes objects from Dataset class.
        Args:
            dataset_path: (str) relative path to dataset directory;
            trainable: (bool) specifies if the dataset will be used for training (True) or validation only (False);
        """

        # States that the dataset's csv files were not loaded yet
        self.is_loaded = False

        # Path to dataset's images
        self.import_dir   = import_dir
        
        # Gets the path to this dataset's CSV file 
        # and the JSON containing its metadata
        self.meta_dir  = os.path.join( ".", "metadata" )
        self.csv_path  = os.path.join( self.meta_dir, f"{name}_data.csv" )
        self.json_path = os.path.join( self.meta_dir, f"{name}_data.json" )

        # Verifies if the dataset's CSV/JSON files exist
        self.check_dataset()

        # Opening JSON file
        with open( self.json_path ) as json_file:
            self.metadata = json.load(json_file)
        
        # Columns to input data's path and output labels
        self.input_col = self.metadata["input_col"]
        self.output_col = self.metadata["output_col"]
        
        # Gets the original number of pneumonia samples
        pneumonia_samples = self.metadata["num_samples"]["total"]["Pneumonia"]
        
        # Wether the samples of pneumonia class should be kept or not
        # A different sufix is applied for each case
        sufix = self.set_class_remap(keep_pneumonia)
        
        # Registers the original dataset name
        self.orig_name = self.metadata["name"]
        
        # Only datasets with pneumonia samples have the sufix added
        self.name = self.metadata["name"]
        if pneumonia_samples > 0:
            self.name = self.name + sufix
        
        # Total number of classes
        self.n_classes = len(self.classes)
        
        # Computes class weights based on self.metadata
        self.set_class_weights()

        return
    
    def check_dataset( self ):
        """ Checks the dataset's directory and looks for its metadata and the
        CSV files with each partition's inputs/outputs. The path to the files 
        are stored inside self.csv_dict, which stores the CSV files' attributes 
        for each partition. """
        for path in [self.csv_path, self.json_path]:
            assert os.path.exists(path), f"\nCouldn't find '{path}' file..."
        return
    
    def set_class_remap(self, keep_pneumonia):
        
        # Wether the samples of pneumonia class should be kept or not
        if keep_pneumonia:
            # If so, Pneumonia and Normal samples are remapped to "neg_COVID-19"
            # and COVID-19 label is renamed to "pos_COVID-19"
            self.label_remap = { "Normal"   : "neg_COVID-19", 
                                 "COVID-19" : "pos_COVID-19", 
                                 "Pneumonia": "neg_COVID-19" }

            self.classes = { "neg_COVID-19": 0, "pos_COVID-19": 1 }
            
            # Iterates through partitions in self.metadata["num_samples"]
            for part in self.metadata["num_samples"].keys():
                # Removes COVID-19 and gets the n° of positive samples
                pos_samples = self.metadata["num_samples"][part].pop("COVID-19")
                
                # Removes Normal/Pneumonia and gets the n° of negative samples
                neg_samples = self.metadata["num_samples"][part].pop("Pneumonia")
                neg_samples += self.metadata["num_samples"][part].pop("Normal")
                
                # Adds new entries to the metadata dict
                self.metadata["num_samples"][part]["neg_COVID-19"] = neg_samples
                self.metadata["num_samples"][part]["pos_COVID-19"] = pos_samples
                
            
            # Different sufix to indicate that samples were remapped
            sufix = "_remapped"
            
        else:
            # Else, Pneumonia samples are droped and the other labels are kept
            self.classes = { "Normal": 0, "COVID-19": 1 }
            self.label_remap = None
            
            # Iterates through partitions in self.metadata["num_samples"]
            for part in self.metadata["num_samples"].keys():
                # Removes pneumonia and gets the n° of dropped samples
                n_dropped = self.metadata["num_samples"][part].pop("Pneumonia")
                
                # Reduce the total number of samples by the ones dropped
                self.metadata["num_samples"][part]["Total"] -= n_dropped
            
            # Different sufix to indicate that samples were discarted
            sufix = "_dropped"
        
        return sufix
    
    def multiclass2binary( self, df ):

        if self.label_remap is None:
            df = df[ df[self.output_col] != "Pneumonia"].reset_index(drop = True)
            return df
        
        df[self.output_col] = df.apply( lambda row: self.label_remap[row[self.output_col]], axis = 1)
        
        return df
    
    def is_df_matched_to_metadata(self):
        if not self.is_loaded:
            print("\nDataframes were not loaded yet...")
            return False
        
        # Lists partitions and classes
        partitions = ["train", "val", "test"]
        classes = [k for k in self.classes.keys()]
        
        # Checks if the number of samples in each partition from the CSV file
        # matches the values from the metadata's JSON file
        for part in partitions:
            # Loads a remapped dataframe
            df = self.get_dataframe(part)
            for clss in classes:
                json_val = self.metadata["num_samples"][part][clss]
                csv_val  = len(df[df[self.output_col] == clss])
                
                if json_val != csv_val:
                    print(f"\t'{self.name}': mismatch ({json_val}!={csv_val})",
                          f"for partition '{part}' and class '{clss}'")
                    return False
                
        return True
    
    def load_dataframes( self, reload = False ):

        if (self.is_loaded) and (not reload):
            return
        
        df = pd.read_csv( self.csv_path, sep = ";" )
        df = self.multiclass2binary(df)

        # Dictionary to store each partition's DataFrame
        self.df_dict  = {}

        # For each considered partition
        for partition in ["train", "val", "test"]:

            # Filters only rows from the current partition and resets the index
            partition_df = df[df["partition"] == partition].copy(deep = True)
            partition_df.reset_index(drop = True, inplace = True)

            # Adds partition DataFrame to df_dict
            self.df_dict[partition] = partition_df
        
        # Updates 'is_loaded' status
        self.is_loaded = True
        
        # Compares CSV data to JSON metadata to make sure the values match
        assert self.is_df_matched_to_metadata(), "".join(["Unable to match",
                  f" sample count from CSV and JSON files for '{self.name}'"])
           
        
        return

    def set_class_weights( self ):
        
        # Gets the n° of samples in total and for each class for training
        train_info = self.metadata["num_samples"]["train"].copy()
        
        # Drops the total number of samples
        train_info.pop("Total")
        
        # Gets the max amount of samples for a class in the train partition
        max_sample_count = np.max(list( train_info.values() ))
        
        self.class_weights = {}
        for clss, current_samples in enumerate(train_info.values()):
            # If there are samples of the current class
            if current_samples > 0:
                # Applies max_sample_count/current_samples as class weight
                self.class_weights[clss] = max_sample_count / current_samples
            
            # Otherwise
            else:
                # Applies 1 to avoid division by 0
                self.class_weights[clss] = 1

        return
    
    def get_relative_path( self ):
        return self.import_dir

    def get_dataframe( self, partition ):
        return self.df_dict[partition].copy(deep = True)
    
    def get_num_samples( self, partition, class_label = "Total" ):
        partition = partition.lower()
        if partition == "validation": 
            partition = "val"
        return self.metadata["num_samples"][partition][class_label]
    
    def get_num_steps( self, partition, batchsize, sampling = None ):
        # If undersampling is not being applied
        if sampling is None:
            # Returns the maximum amount of batches produceable
            return int(np.ceil(self.get_num_samples(partition)/float(batchsize)))
        
        # Otherwise, computes how many batches are possible 
        # for the chosen sampling method
        partition_info = self.metadata["num_samples"][partition]
        max_samples = np.max(list( partition_info.values() ))
        min_samples = np.min(list( partition_info.values() ))
        ref_samples = max_samples if sampling == "oversampling" else min_samples
        
        return int(np.ceil(ref_samples/float(batchsize/self.n_classes)))
        
def load_datasets( import_dir, train_dataset_name, use_val_data, 
                   keep_pneumonia ):

    # List of all available datasets
    csv_paths = glob.glob(os.path.join("metadata", "*_data.csv"))
    dataset_list = [os.path.basename(p).split("_data")[0] for p in csv_paths]

    print("\nChecking datasets:")
    if keep_pneumonia:
        print("\tRenaming 'COVID-19' labels to 'pos_COVID-19' for positive samples...")
        print("\tRemapping 'Normal' and 'Pneumonia' to 'neg_COVID-19' for negative samples...")
        
    else:
        print("\tDropping all samples from class 'Pneumonia'...")
    
    train_dataset = None
    val_dataset_list = []
    for dataset_name in dataset_list:
        
        # Builds object to handle the current dataset
        dataset_obj = Dataset( import_dir, name = dataset_name, 
                               keep_pneumonia = keep_pneumonia )

        # Check wether this dataset will be used for training or validation
        if (dataset_name == train_dataset_name):
            # Sets object as train dataset
            train_dataset = dataset_obj

        elif not use_val_data:
            # Else, appends dataset to validation list
            val_dataset_list.append( dataset_obj )
    
    if len(val_dataset_list) == 0:
        val_dataset_list = None
        
    else:
        # Sorts validation data from largest dataset to smallest
        val_dataset_list = sorted(val_dataset_list, reverse = True, 
                             key = lambda dset: dset.get_num_samples("total"))
    
    return train_dataset, val_dataset_list
