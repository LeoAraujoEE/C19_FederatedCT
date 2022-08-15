import os
import random
import numpy as np
import tensorflow as tf

class CustomDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, dataset, partition, hyperparameters, aug_dict = None, undersample = False, shuffle = True, seed = None):

        # Generates random seed if not provided
        if seed is None:
            seed = np.random.randint( 0, 255)

        # Sets the given seed to random, numpy and tensorflow for reproducibility
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # If Random Undersampling should be applied to balance the batches
        self.undersample = undersample

        # Extracts the targeted shape
        inputH = hyperparameters["input_height"]
        inputW = hyperparameters["input_width"]
        inputC = hyperparameters["input_channels"]
        
        # Defines other metadata based on the given parameters 
        self.batch_size = hyperparameters["batchsize"]
        self.input_size = (inputH, inputW, inputC)
        self.shuffle = shuffle

        # Gets the dataframe with input/output paths 
        df = dataset.get_dataframe(partition)
        
        # Gets the relative path to the dataset
        image_dir = f"{inputH}x{inputW}"
        data_path = dataset.get_relative_path()
        self.import_dir = os.path.join(data_path, image_dir)
        
        # If the targeted size is unavailable,
        # samples are resized from 512x512 images
        if not os.path.exists(self.import_dir):
            self.import_dir = os.path.join(data_path, "512x512")    

        # Gets the names of the input and output columns in the dataf ame
        self.X_col = dataset.input_col
        self.Y_col = dataset.output_col

        # Identifies each individual class from given labels
        self.unq_labels = [ clss for clss in dataset.classes.keys() ]
        self.label2class_dict = { label: clss for clss, label in enumerate(self.unq_labels) }
        self.class2label_dict = { clss: label for clss, label in enumerate(self.unq_labels) }

        # Gets the number of samples and the number of classes in the dataset
        self.n = len(df)
        self.n_classes = len(self.unq_labels)

        if self.n_classes != len(self.unq_labels):
            print( f"\nOnly {dataset.n_classes} of {self.n_classes} classes are represented in this dataset..." )

        self.set_dataframes( df )
            
        # Saves the augmentation dict whithin the generator class
        self.set_datagen( hyperparameters, aug_dict )

        return
    
    @staticmethod
    def custom_preprocess_input(model_input):
        model_input = model_input.astype(np.float32)
        model_input /= 127.5
        model_input -= 1
        return model_input

    def set_dataframes( self, df ):
        self.dfs = []

        if not self.undersample:
            self.dfs.append( df )
            return
        
        self.batch_size = self.batch_size // 2

        for label in self.label2class_dict.keys():
            sub_df = df[ df[self.Y_col] == label ].copy(deep = True)
            sub_df.reset_index(drop = True, inplace = True)
            self.dfs.append( sub_df )
        return

    def set_datagen(self, hyperparameters, augmentation_dict):
        
        if (augmentation_dict is None) or (not hyperparameters["augmentation"]):
            # Transformations for ImageDataGenerator
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale                = 1.,
                                                                      preprocessing_function = CustomDataGenerator.custom_preprocess_input
                                                                     )
        
        else:
            # Zoom range for data augmentation
            zoom_range = (1. - augmentation_dict["zoom_in"], 1. + augmentation_dict["zoom_out"])

            # brightness range for data augmentation
            brightness_range = (1. - augmentation_dict["brightness"], 1. + augmentation_dict["brightness"])

            # Transformations for ImageDataGenerator
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale                = 1.,
                                                                      rotation_range         = augmentation_dict["rotation"],
                                                                      shear_range            = augmentation_dict["shear"],
                                                                      brightness_range       = brightness_range,
                                                                      width_shift_range      = augmentation_dict["horizontal_translation"],
                                                                      height_shift_range     = augmentation_dict["vertical_translation"],
                                                                      zoom_range             = zoom_range,
                                                                      horizontal_flip        = augmentation_dict["horizontal_flip"],
                                                                      vertical_flip          = augmentation_dict["vertical_flip"],
                                                                      fill_mode              = augmentation_dict["fill_mode"],
                                                                      cval                   = augmentation_dict["constant_val"], 
                                                                      preprocessing_function = CustomDataGenerator.custom_preprocess_input
                                                                     )

        input_H, input_W, input_C = self.input_size
        color_mode = "grayscale" if input_C == 1 else "rgb"

        self.generators = []
        for sub_df in self.dfs:
            # Adjusts labels in the generator's dataframe to keep desired class order (0: not COVID, 1: COVID)
            gen_df = sub_df.copy(deep = True)
            gen_df[self.Y_col] = gen_df.apply( lambda r: int(self.label2class_dict[r[self.Y_col]]), axis = 1 )

            # Builds a data generator object
            generator = datagen.flow_from_dataframe( dataframe = gen_df, x_col = self.X_col, y_col = self.Y_col, 
                                                     target_size = (input_H, input_W), batch_size = self.batch_size, 
                                                     class_mode = "raw", directory = self.import_dir, 
                                                     color_mode = color_mode, shuffle = self.shuffle, seed = self.seed 
                                                     )
            self.generators.append(generator)

        return 
    
    def merge_dfs(self):
        n_samples_from_df = np.min([len(df) for df in self.dfs])

        base_idxs = [i for i in range(n_samples_from_df)]
        neg_idxs = [ int(self.batch_size * np.floor(i / self.batch_size) + i) for i in base_idxs]
        pos_idxs = [ int(self.batch_size * np.ceil((i+1) / self.batch_size) + i) for i in base_idxs]

        df_neg = self.dfs[0][:n_samples_from_df]
        df_neg.index = neg_idxs

        df_pos = self.dfs[1][:n_samples_from_df]
        df_pos.index = pos_idxs

        import pandas as pd
        df = pd.concat( [df_neg, df_pos] )
        df.sort_index( inplace = True )
        return df

    def get_labels(self):
        # Gets all labels in the dataframe as their corresponding class numbers
        # Won't work if self.shuffle == True or if resample is being used
        if self.shuffle:
            print("\nDatagen is set to Shuffle, class labels are probably inconsistent...")
        
        if self.undersample:
            clss_labels = [self.label2class_dict[l] for l in self.merge_dfs()[self.Y_col].to_list()]
            return clss_labels

        clss_labels = [self.label2class_dict[l] for l in self.dfs[0][self.Y_col].to_list()]
        return clss_labels
        
    def get_fnames(self):
        # Gets all labels in the dataframe as their corresponding class numbers
        # Won't work if self.shuffle == True or if resample is being used
        if self.shuffle:
            print("\nDatagen is set to Shuffle, filenames are probably inconsistent...")
        
        if self.undersample:
            fnames = [os.path.basename(p) for p in self.merge_dfs()[self.X_col].to_list()]
            return fnames

        fnames = [os.path.basename(p) for p in self.dfs[0][self.X_col].to_list()]
        return fnames
    
    def __getitem__(self, index):
        X = []
        Y = []
        for generator in self.generators:
            batch = generator.__getitem__(index)
            X.append( batch[0] )
            Y.append( batch[1] )

        X = np.concatenate(X, axis = 0).astype(np.float32)
        Y = np.concatenate(Y, axis = 0).astype(np.float32)
        return X, Y
    
    def __len__(self):
        # Returns the number of batches the generator can produce
        if not self.undersample:
            return self.generators[0].__len__()
        return np.min([gen.__len__() for gen in self.generators])