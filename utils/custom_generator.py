import os
import random
import numpy as np
import tensorflow as tf

class CustomDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, dataset, partition, hyperparameters, aug_dict = {}, undersample = False, shuffle = True, seed = None):

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

        # Gets the dataframe with input/output paths 
        # and the relative path to the dataset
        df = dataset.get_dataframe(partition)
        self.import_dir = dataset.get_relative_path()

        # Gets the names of the input and output columns in the dataf ame
        self.X_col = getattr( dataset, "input_col")
        self.Y_col = getattr( dataset, "output_col")

        # Identifies each individual class from given labels
        self.unq_labels = [ clss for clss in getattr( dataset, "classes" ).keys() ]
        self.label2class_dict = { label: clss for clss, label in enumerate(self.unq_labels) }
        self.class2label_dict = { clss: label for clss, label in enumerate(self.unq_labels) }

        # Gets the number of samples and the number of classes in the dataset
        self.n = len(df)
        self.n_classes = len(self.unq_labels)

        if self.n_classes != len(self.unq_labels):
            print( "\nOnly {} of {} classes are represented in this dataset...".format( getattr(dataset, "n_classes"), self.n_classes ) )

        # Defines other metadata based on the given parameters
        self.batch_size = hyperparameters["batchsize"]
        self.input_size = hyperparameters["input_size"]
        self.shuffle = shuffle

        self.set_dataframes( df )
            
        # Saves the augmentation dict whithin the generator class
        self.set_datagen( hyperparameters, aug_dict )

        return
    
    def fill_aug_dict(self, hyperparameters, augmentation_dict):

        # List of data augmentation parameters
        base_aug_dict = { "zoom":                    0.00,     # Max zoom in/zoom out
                          "shear":                   00.0,     # Max random shear
                          "rotation":                00.0,     # Max random rotation
                          "shear_range":             00.0,
                          "vertical_translation":    0.00,     # Max vertical translation
                          "horizontal_translation":  0.00,     # Max horizontal translation
                          "vertical_flip":          False,     # Allow vertical flips  
                          "horizontal_flip":        False,     # Allow horizontal flips    
                          "brightness":              0.00,     # Brightness adjustment range
                          "channel_shift":           00.0,     # Random adjustment to random channel
                          "constant_val":            00.0,
                          "fill_mode":          "constant"
                      }

        if not hyperparameters["augmentation"]:
            return base_aug_dict

        for k, v in base_aug_dict.items():
            if not k in augmentation_dict.keys():
                augmentation_dict[k] = v

        return augmentation_dict

    @staticmethod
    def get_preprocessing_function(hyperparameters):
        # If a preprocess_func isnt being used, no function is returned
        if (not hyperparameters["preprocess_func"]):
            return None

        architecture_name = hyperparameters["architecture"].lower()

        if "xception" in architecture_name:
            return tf.keras.applications.xception.preprocess_input

        if ("resnet" in architecture_name) and ("v2" in architecture_name):
            return tf.keras.applications.resnet_v2.preprocess_input

        if ("mobilenet" in architecture_name) and ("v2" in architecture_name):
            return tf.keras.applications.mobilenet_v2.preprocess_input

        if "densenet" in architecture_name:
            return tf.keras.applications.densenet.preprocess_input

        if architecture_name in ["inception_v3", "inceptionv3"]:
            return tf.keras.applications.inception_v3.preprocess_input

        if architecture_name in ["inception_resnet_v2", "inception_resnetv2", "inceptionresnet_v2", "inceptionresnetv2"]:
            return tf.keras.applications.inception_resnet_v2.preprocess_input

        if architecture_name in ["vgg_16", "vgg16"]:
            return tf.keras.applications.vgg16.preprocess_input

        if architecture_name in ["vgg_19", "vgg19"]:
            return tf.keras.applications.vgg19.preprocess_input

        if "efficientnet" in architecture_name:
            return tf.keras.applications.efficientnet.preprocess_input

        raise ValueError("\nUnknown architecture...")

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

        augmentation_dict = self.fill_aug_dict(hyperparameters, augmentation_dict)

        # Datagen
        # Factor to normalize inputs, is set to 1. if a keras.applications preprocess_func is used
        rescale_factor  = 1. if hyperparameters["preprocess_func"] else 1./255.

        # The keras.applications preprocess_func used, defaults to None if no function is selected
        preprocess_func = self.get_preprocessing_function(hyperparameters)

        # Zoom range for data augmentation
        zoom_range = ( 1. - augmentation_dict["zoom"], 1. + augmentation_dict["zoom"])

        # brightness range for data augmentation
        brightness_range = ( 1. - augmentation_dict["brightness"], 1. + augmentation_dict["brightness"])

        # Transformations for ImageDataGenerator
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale                =    rescale_factor,
                                                                  rotation_range         =    augmentation_dict["rotation"],
                                                                  shear_range            =    augmentation_dict["shear_range"],
                                                                  brightness_range       =    brightness_range,
                                                                  width_shift_range      =    augmentation_dict["horizontal_translation"],
                                                                  height_shift_range     =    augmentation_dict["vertical_translation"],
                                                                  zoom_range             =    zoom_range,
                                                                  horizontal_flip        =    augmentation_dict["horizontal_flip"],
                                                                  vertical_flip          =    augmentation_dict["vertical_flip"],
                                                                  fill_mode              =    augmentation_dict["fill_mode"],
                                                                  cval                   =    augmentation_dict["constant_val"], 
                                                                  preprocessing_function = preprocess_func
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
        n_samples_from_df = len(self) * self.batch_size

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
            print("\nDatagen is set to Random Undersample, filenames are probably inconsistent...")
            fnames = [os.path.basename(p) for p in self.merge_dfs()[self.X_col].to_list()]
            return fnames

        fnames = [os.path.basename(p) for p in self.dfs[0][self.X_col].to_list()]
        return fnames
    
    def __getitem__(self, index):
        # item = self.generator.__getitem__(index)
        # print( f"item[0].shape: {item[0].shape}, item[1].shape: {item[1].shape}, item[1]: {item[1]}" )
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
        # max_batces = int(np.ceil(self.n / float(self.batch_size)))
        if not self.undersample:
            return self.generators[0].__len__()
        return np.min([gen.__len__() for gen in self.generators])
    

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, dataset, partition, hyperparameters, aug_dict = {}, shuffle = True, seed = None):

        # Generates random seed if not provided
        if seed is None:
            seed = np.random.randint( 0, 255)

        # Sets the given seed to random, numpy and tensorflow for reproducibility
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Gets the dataframe with input/output paths 
        # and the relative path to the dataset
        self.df = dataset.get_dataframe(partition)
        self.import_dir = dataset.get_relative_path()

        # Gets the names of the input and output columns in the dataf ame
        self.X_col = getattr( dataset, "input_col")
        self.Y_col = getattr( dataset, "output_col")

        # Identifies each individual class from given labels
        self.unq_labels = [ clss for clss in getattr( dataset, "classes" ).keys() ]
        self.label2class_dict = { label: clss for clss, label in enumerate(self.unq_labels) }
        self.class2label_dict = { clss: label for clss, label in enumerate(self.unq_labels) }

        # Gets the number of samples and the number of classes in the dataset
        self.n = dataset.get_num_samples(partition)
        self.n_classes = len(self.unq_labels)

        if self.n_classes != len(self.unq_labels):
            print( "\nOnly {} of {} classes are represented in this dataset...".format( getattr(dataset, "n_classes"), self.n_classes ) )

        # Defines other metadata based on the given parameters
        self.batch_size = hyperparameters["batchsize"]
        self.input_size = hyperparameters["input_size"]
        self.shuffle = shuffle
            
        # Saves the augmentation dict whithin the generator class
        self.set_datagen( hyperparameters, aug_dict )

        return
    
    def fill_aug_dict(self, hyperparameters, augmentation_dict):

        # List of data augmentation parameters
        base_aug_dict = { "zoom":                    0.00,     # Max zoom in/zoom out
                          "shear":                   00.0,     # Max random shear
                          "rotation":                00.0,     # Max random rotation
                          "shear_range":             00.0,
                          "vertical_translation":    0.00,     # Max vertical translation
                          "horizontal_translation":  0.00,     # Max horizontal translation
                          "vertical_flip":          False,     # Allow vertical flips  
                          "horizontal_flip":        False,     # Allow horizontal flips    
                          "brightness":              0.00,     # Brightness adjustment range
                          "channel_shift":           00.0,     # Random adjustment to random channel
                          "constant_val":            00.0,
                          "fill_mode":          "constant"
                      }

        if not hyperparameters["augmentation"]:
            return base_aug_dict

        for k, v in base_aug_dict.items():
            if not k in augmentation_dict.keys():
                augmentation_dict[k] = v

        return augmentation_dict

    @staticmethod
    def get_preprocessing_function(hyperparameters):
        # If a preprocess_func isnt being used, no function is returned
        if (not hyperparameters["preprocess_func"]):
            return None

        architecture_name = hyperparameters["architecture"].lower()

        if "xception" in architecture_name:
            return tf.keras.applications.xception.preprocess_input

        if ("resnet" in architecture_name) and ("v2" in architecture_name):
            return tf.keras.applications.resnet_v2.preprocess_input

        if ("mobilenet" in architecture_name) and ("v2" in architecture_name):
            return tf.keras.applications.mobilenet_v2.preprocess_input

        if "densenet" in architecture_name:
            return tf.keras.applications.densenet.preprocess_input

        if architecture_name in ["inception_v3", "inceptionv3"]:
            return tf.keras.applications.inception_v3.preprocess_input

        if architecture_name in ["inception_resnet_v2", "inception_resnetv2", "inceptionresnet_v2", "inceptionresnetv2"]:
            return tf.keras.applications.inception_resnet_v2.preprocess_input

        if architecture_name in ["vgg_16", "vgg16"]:
            return tf.keras.applications.vgg16.preprocess_input

        if architecture_name in ["vgg_19", "vgg19"]:
            return tf.keras.applications.vgg19.preprocess_input

        if "efficientnet" in architecture_name:
            return tf.keras.applications.efficientnet.preprocess_input

        raise ValueError("\nUnknown architecture...")

    def set_datagen(self, hyperparameters, augmentation_dict):

        augmentation_dict = self.fill_aug_dict(hyperparameters, augmentation_dict)

        # Datagen
        # Factor to normalize inputs, is set to 1. if a keras.applications preprocess_func is used
        rescale_factor  = 1. if hyperparameters["preprocess_func"] else 1./255.

        # The keras.applications preprocess_func used, defaults to None if no function is selected
        preprocess_func = self.get_preprocessing_function(hyperparameters)

        # Zoom range for data augmentation
        zoom_range = ( 1. - augmentation_dict["zoom"], 1. + augmentation_dict["zoom"])

        # brightness range for data augmentation
        brightness_range = ( 1. - augmentation_dict["brightness"], 1. + augmentation_dict["brightness"])

        # Transformations for ImageDataGenerator
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale                =    rescale_factor,
                                                                  rotation_range         =    augmentation_dict["rotation"],
                                                                  shear_range            =    augmentation_dict["shear_range"],
                                                                  brightness_range       =    brightness_range,
                                                                  width_shift_range      =    augmentation_dict["horizontal_translation"],
                                                                  height_shift_range     =    augmentation_dict["vertical_translation"],
                                                                  zoom_range             =    zoom_range,
                                                                  horizontal_flip        =    augmentation_dict["horizontal_flip"],
                                                                  vertical_flip          =    augmentation_dict["vertical_flip"],
                                                                  fill_mode              =    augmentation_dict["fill_mode"],
                                                                  cval                   =    augmentation_dict["constant_val"], 
                                                                  preprocessing_function = preprocess_func
                                                                 )

        input_H, input_W, input_C = self.input_size
        color_mode = "grayscale" if input_C == 1 else "rgb"

        gen_df = self.df.copy(deep = True)
        gen_df[self.Y_col] = gen_df.apply( lambda r: str(self.label2class_dict[r[self.Y_col]])+"_"+r[self.Y_col], axis = 1 )
        self.generator = datagen.flow_from_dataframe( dataframe = gen_df, x_col = self.X_col, y_col = self.Y_col, 
                                                    target_size = (input_H, input_W), batch_size = self.batch_size, 
                                                    class_mode = "categorical", directory = self.import_dir, 
                                                    color_mode = color_mode, shuffle = self.shuffle, seed = self.seed 
                                                    )
                                        
        # print("This is flow_from_dataframe.class_indices:", self.generator.class_indices)
        # print("This is self.label2class_dict:", self.label2class_dict)

        return 

    def get_labels(self):
        # Gets all labels in the dataframe as their corresponding class numbers
        # Won't work if self.shuffle == True
        if self.shuffle:
            print("\nDatagen is set to Shuffle, class labels are probably inconsistent...")
        clss_labels = [self.label2class_dict[l] for l in self.df[self.Y_col].to_list()]
        return clss_labels
        
    def get_fnames(self):
        # Gets all labels in the dataframe as their corresponding class numbers
        # Won't work if self.shuffle == True
        if self.shuffle:
            print("\nDatagen is set to Shuffle, class labels are probably inconsistent...")
        fnames = [os.path.basename(p) for p in self.df[self.X_col].to_list()]
        return fnames
    
    def __getitem__(self, index):
        # item = self.generator.__getitem__(index)
        # print( f"item[0].shape: {item[0].shape}, item[1].shape: {item[1].shape}, item[1]: {item[1]}" )
        return self.generator.__getitem__(index)
    
    def __len__(self):
        # Returns the number of batches the generator can produce
        # max_batces = int(np.ceil(self.n / float(self.batch_size)))
        return self.generator.__len__()