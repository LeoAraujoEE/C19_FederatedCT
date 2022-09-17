import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

class CustomDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, dataset, partition, hyperparameters, aug_dict = None, sampling = None, shuffle = True, seed = None):

        # Generates random seed if not provided
        if seed is None:
            seed = np.random.randint( 0, 255)

        # Sets the given seed to random, numpy and tensorflow for reproducibility
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # How the sampling should be done, can be either:
        # None, for regular sampling from ImageDataGenerator
        # oversampling, for random oversampling
        # undersampling, for random undersampling
        self.sampling = None
        if isinstance(sampling, str) and sampling.lower() in ["oversampling", "undersampling"]:
            self.sampling = sampling.lower()

        # Extracts the targeted shape
        inputH = hyperparameters["input_height"]
        inputW = hyperparameters["input_width"]
        inputC = hyperparameters["input_channels"]
        
        # Defines other metadata based on the given parameters 
        self.batch_size = hyperparameters["batchsize"]
        self.input_size = (inputH, inputW, inputC)
        self.shuffle = shuffle

        # Gets the dataframe with input/output paths 
        self.base_df = dataset.get_dataframe(partition)
        
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
        self.n = len(self.base_df)
        self.n_classes = len(self.unq_labels)

        if self.n_classes != len(self.unq_labels):
            print( f"\nOnly {dataset.n_classes} of {self.n_classes} classes are represented in this dataset..." )
            
        # Saves the augmentation dict whithin the generator class
        self.set_datagen_transformations( hyperparameters, aug_dict )
        self.set_generator()

        return
    
    @staticmethod
    def custom_preprocess_input(model_input):
        model_input = model_input.astype(np.float32)
        model_input /= 127.5
        model_input -= 1
        return model_input

    def get_sampled_df(self, sub_df, ref_samples):
        if ref_samples == len(sub_df):
            return sub_df
        
        if self.sampling == "undersampling":
            # Undersamples larger dfs to match ref_samples
            sub_df = sub_df.sample(n = ref_samples, replace = False)
        
        if self.sampling == "oversampling":
            # Upsamples smaller dfs to match ref_samples
            sub_df = sub_df.sample(n = ref_samples, replace = True)
            
        sub_df.reset_index(drop = True, inplace = True)
        return sub_df

    def resample_dataframe( self, src_df ):
        if self.sampling is None:
            return src_df
        
        class_dfs = []
        class_samples = []
        for label in self.label2class_dict.keys():
            class_dfs.append(src_df[src_df[self.Y_col] == label].copy(deep = True))
            class_samples.append( len(class_dfs[-1]) )
        
        max_samples = np.max(class_samples)
        min_samples = np.min(class_samples)
        ref_samples = max_samples if self.sampling == "oversampling" else min_samples
        
        # Resamples each class' dataframe based on the sampling method
        class_dfs = [self.get_sampled_df(df, ref_samples) for df in class_dfs]
        dst_df = pd.concat(class_dfs, axis = 0, ignore_index = True)
        
        if self.shuffle:
            dst_df = dst_df.sample(frac=1).reset_index(drop=True)
        
        return dst_df
    
    def set_datagen_transformations(self, hyperparameters, augmentation_dict):
        
        if (augmentation_dict is None) or (not hyperparameters["augmentation"]):
            # Transformations for ImageDataGenerator
            self.datagen = tf.keras.preprocessing.image.ImageDataGenerator( rescale = 1.,
                                                                      preprocessing_function = CustomDataGenerator.custom_preprocess_input 
                                                                      )
        
        else:
            # Zoom range for data augmentation
            zoom_range = ( 1. - augmentation_dict["zoom_in"], 
                           1. + augmentation_dict["zoom_out"] )

            # brightness range for data augmentation
            brightness_range = ( 1. - augmentation_dict["brightness"], 
                                 1. + augmentation_dict["brightness"] )

            # Transformations for ImageDataGenerator
            self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale           = 1.,
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
        return

    def set_generator(self):

        self.df = self.resample_dataframe(self.base_df)
        
        input_H, input_W, input_C = self.input_size
        color_mode = "grayscale" if input_C == 1 else "rgb"

        # Adjusts labels in the generator's dataframe to keep desired class order (0: not COVID, 1: COVID)
        gen_df = self.df.copy(deep = True)
        gen_df[self.Y_col] = gen_df.apply( lambda r: int(self.label2class_dict[r[self.Y_col]]), axis = 1 )

        # Builds a data generator object
        self.generator = self.datagen.flow_from_dataframe(dataframe = gen_df, x_col = self.X_col, y_col = self.Y_col, 
                                target_size = (input_H, input_W), batch_size = self.batch_size, class_mode = "raw",
                                directory = self.import_dir, color_mode = color_mode, shuffle = False, 
                                seed = self.seed)
        return 
    
    def on_epoch_end(self):
        # On the end of each epoch, if applying random undersampling or 
        # oversampling, resets the generator to choose new samples 
        # for the next epoch
        if not self.sampling is None:
            self.set_generator()
        return 

    def get_labels(self):
        # Gets all labels in the dataframe as their corresponding class numbers
        return [self.label2class_dict[l] for l in self.df[self.Y_col].to_list()]
        
    def get_fnames(self):
        # Gets all labels in the dataframe as their corresponding class numbers
        return [os.path.basename(p) for p in self.df[self.X_col].to_list()]
    
    def __getitem__(self, index):
        return self.generator.__getitem__(index)
    
    def __len__(self):
        # Returns the number of batches the generator can produce
        return len(self.generator)