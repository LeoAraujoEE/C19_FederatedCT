import os
import numpy as np
import pandas as pd

class Dataset():
    def __init__( self, import_dir, folder, input_col, output_col, trainable = False, keep_pneumonia = False ):
        """ Initializes objects from Dataset class.
        Args:
            dataset_path: (str) relative path to dataset directory;
            trainable: (bool) specifies if the dataset will be used for training (True) or validation only (False);
        """

        # States that the dataset's csv files were not loaded yet
        self.is_loaded = False

        # Builds dataset_path from import_dir and folder
        dataset_path  = os.path.join( import_dir, folder )
        metadata_path = os.path.join( import_dir, folder+"_data.csv" )
        
        # Wether the samples of pneumonia class should be kept or not
        if keep_pneumonia:
            # If so, Pneumonia and Normal samples are remapped to "neg_COVID-19"
            # and COVID-19 label is renamed to "pos_COVID-19"
            self.label_remap = { "Normal"   : "neg_COVID-19", 
                                 "COVID-19" : "pos_COVID-19", 
                                 "Pneumonia": "neg_COVID-19" }

            classes = { "neg_COVID-19": 0, "pos_COVID-19": 1 }

            # Sets the dataset name based on wether pneumonia samples were kept or not
            # also drops the ".org" for Radiopaedia's dataset name
            name = "{}_remapped".format(folder.split(".")[0])
            
        else:
            # Else, Pneumonia samples are droped and the other labels are kept
            classes = { "Normal": 0, "COVID-19": 1 }
            self.label_remap = None

            # Sets the dataset name based on wether pneumonia samples were kept or not
            # also drops the ".org" for Radiopaedia's dataset name
            name = "{}_dropped".format(folder.split(".")[0])
        
        # Register the dataset information to class variables
        self.name = name                   # Dataset's name
        self.classes = classes             # Dict of class names and labels
        self.n_classes = len(classes)      # Total number of classes
        self.trainable = trainable         # If train/val data will be used
        self.input_col = input_col         # Column to input data's path
        self.output_col = output_col       # Column to output labels
        self.import_dir = import_dir       # Path to data dir
        self.dataset_path = dataset_path   # Path to dataset's images
        self.metadata_path = metadata_path # Path to dataset's metadata

        # Verifies if the partition's CSV files and the dataset's JSON file exist
        self.check_dataset()

        return
    
    def check_dataset( self ):
        """ Checks the dataset's directory and looks for its metadata and the CSV files with each partition's inputs/outputs.
        The path to the files are stored inside self.csv_dict, which stores the CSV files' attributes for each partition.
        """
        
        print("Looking for '{}' dataset's CSV file...".format(self.name), end="\r")
        assert os.path.exists(self.metadata_path), "\nCouldn't find '{}' file...".format(self.metadata_path)
        print( "Found '{}' dataset's CSV file at '{}' path...".format(self.name, self.metadata_path) )

        return
    
    def multiclass2binary( self, df ):

        if self.label_remap is None:
            print("\nDropping all samples from class 'Pneumonia'...")
            df = df[ df[self.output_col] != "Pneumonia"].reset_index(drop = True)
            return df
        
        print("\nRenaming 'COVID-19' labels to 'pos_COVID-19' for positive samples...")
        print("Remapping 'Normal' and 'Pneumonia' to 'neg_COVID-19' for negative samples...")
        df[self.output_col] = df.apply( lambda row: self.label_remap[row[self.output_col]], axis = 1)
        
        return df
    
    def load_dataframes( self, reload = False ):

        if (self.is_loaded) and (not reload):
            print("\nCSV file is already loaded as pd.Dataframe...")
            return
        
        print("\nLoading CSV file for '{}' as pd.DataFrame:".format(self.name), end="\r")
        metadata_df = pd.read_csv( self.metadata_path, sep = ";" )
        metadata_df = self.multiclass2binary(metadata_df)
        print("Loaded CSV file as pd.DataFrame...")

        # Dictionary to store objects related to each partition
        self.csv_dict  = {}

        # List of partitions to check
        partition_list = ["train", "val", "test"] if self.trainable else ["test"]

        # For each considered partition
        print("\nExtracting data for each partition...")
        for partition in partition_list:
            # Creates a dictionary for its CSV files attributes
            partition_dict = { "df": None, "n_samples": None }

            print("\tExtracting '{}' partition's metadata...".format(partition), end="\r")
            partition_df = metadata_df[ metadata_df["partition"] == partition ].copy(deep = True)
            partition_df.reset_index(drop = True, inplace = True)

            # Loads the CSV and computes the number of samples
            partition_dict["df"] = partition_df
            partition_dict["n_samples"] = len(partition_dict["df"])
            print("\tExtracted '{}' partition's metadata, {} samples found...".format(partition, partition_dict["n_samples"]))

            # Adds partition dict to csv_dict
            self.csv_dict[partition] = partition_dict
        
        self.is_loaded = True

        # If trainable, computes class_weights for this dataset
        # Does not repeat this if the dataframes are being reloaded
        if (self.trainable) and (not hasattr(self, "class_weights")):
            self.compute_class_weights()
        
        return

    def compute_class_weights( self ):
        df = self.csv_dict["train"]["df"]
        class_list = [ clss for clss in self.classes.keys() ]

        counts = [ len(df[df[self.output_col] == clss]) for clss in class_list ]

        weights = [ (np.max( counts ) / cts) if cts > 0 else 1 for cts in counts ]

        self.class_weights = { clss: weight for clss, weight in enumerate(weights) }
        
        # print( "\tComputed class_weights:" )
        # for idx, v in enumerate(self.class_weights.values()):
        #     print( "\t\t[{}] {}: {:.3f}".format(str(idx).zfill(2), class_list[idx], v) )
        # print("\n")

        return
    
    def get_relative_path( self ):
        return self.import_dir

    def get_dataframe( self, partition ):
        return self.csv_dict[partition]["df"]
    
    def get_num_samples( self, partition ):
        partition = partition.lower() if partition.lower() != "validation" else "val"
        return self.csv_dict[partition]["n_samples"]
    
    def get_num_steps( self, partition, batchsize ):
        return int(np.ceil(self.get_num_samples( partition ) / float(batchsize)))


def load_datasets( import_dir, train_dataset, input_col, output_col, keep_pneumonia ):

    # List of all available datasets
    available_datasets = [ "Comp_CNCB_iCTCF_a", "Comp_CNCB_iCTCF_b", "radiopaedia.org",
                           "COVID-CTSet", "COVID-CT-MD", "Comp_LIDC-SB" ]
    
    val_dataset_list = []
    for dataset_name in available_datasets:

        # Check wether this dataset will be used for training or validation
        is_trainable = (dataset_name == train_dataset)
        
        # Builds object to handle the current dataset
        dataset_obj = Dataset( import_dir, folder = dataset_name, input_col = input_col, 
                               output_col = output_col, keep_pneumonia = keep_pneumonia, 
                               trainable = is_trainable )

        if is_trainable:
            # Sets object as train dataset
            train_dataset = dataset_obj

        else:
            # Else, appends dataset to validation list
            val_dataset_list.append( dataset_obj )
    
    return train_dataset, val_dataset_list
