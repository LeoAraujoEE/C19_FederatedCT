import os
import json
import random
import tempfile
import numpy as np
import tensorflow as tf

from utils.architectures.resnet import ResNet
from utils.architectures.densenet import DenseNet
from utils.architectures.xception import Xception
from utils.architectures.mobilenet import MobileNet
from utils.architectures.inception import Inception
from utils.architectures.efficientnet import EfficientNet
from utils.architectures.inception_resnet import InceptionResNet
from tensorflow.python.keras.utils.layer_utils import count_params

class ModelBuilder:

    def __init__( self, model_path, gen_fig = False ):
        # Specifies the relative path to the model's directory
        self.model_path = model_path
        self.model_dir  = os.path.dirname(model_path)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Indicates whether to save a figure of the model or not
        self.gen_fig = gen_fig

        return

    def __call__(self, hyperparameters, seed):
        """ 
        Supported architectures: 
            Resnet         :  18,  34,  50, 101, 152
            Densenet       : 121, 169, 201, 264
            Inception      :  V3,  V4
            InceptionResNet:  V2
            MobileNet      :  V2, V3_Small, V3_Large
            EfficientNet   :  B0, B1, B2, B3, B4, B5, B6, B7
            EfficientNetV2 :  B0, B1, B2, B3,  S,  M,  L
        """
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Creates a model with all layers up to the end of the convolutional base
        model  = self.create_model(hyperparameters)
        
        # Counts the model's parameters
        # trainable_count = int(np.sum([ count_params(l.trainable_weights) for l in model.layers ]))
        # non_trainable_count = int(np.sum([ count_params(l.non_trainable_weights) for l in model.layers ]))
        trainable_count, non_trainable_count = self.count_model_params(model)
        print("\nCreated model with {:,} trainable parameters and {:,} non trainable ones...".format(trainable_count, non_trainable_count))

        # Saves model configs
        json_config = model.to_json()
        config_path = self.model_path.replace(".h5", ".json")

        with open(config_path, "w") as json_file:
            json.dump( json_config, json_file, indent=4 )

        # Generates the model plot if specified
        if self.gen_fig:        
            self.gen_model_as_figure(model)

        return model
    
    @staticmethod
    def count_model_params(model):
        
        # Initializes counters
        trainable, non_trainable = 0, 0
        
        # Iterates layers
        for l in model.layers:
            # Counts parameters
            trainable += count_params(l.trainable_weights)
            non_trainable += count_params(l.non_trainable_weights)
            
        return int(trainable), int(non_trainable)

    def gen_model_as_figure(self, model):
        path = os.path.join( self.model_dir, "model_fig.png" )

        print(f"Saving fig to '{path}'...")
        tf.keras.utils.plot_model( model, to_file = path, show_shapes = True, show_layer_names = True, 
                                   rankdir = "TB", expand_nested = False, dpi = 96 )
        
        return

    def create_model(self, hyperparameters):
        inputH = hyperparameters["input_height"]
        inputW = hyperparameters["input_width"]
        inputC = hyperparameters["input_channels"]
        input_size = (inputH, inputW, inputC)
        
        if "inception_resnet" in hyperparameters["architecture"].lower():
            builder = InceptionResNet()
            model  = builder.get_InceptionResNetV2( input_size, 1, "sigmoid", "avg", 
                             hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                             hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "resnet" in hyperparameters["architecture"].lower():
            builder = ResNet()
            
            if hyperparameters["architecture"].lower() == "custom_resnet18":
                model  = builder.get_ResNet18( input_size, 1, "sigmoid", "avg", 
                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
                
            elif hyperparameters["architecture"].lower() == "custom_resnet34":
                model  = builder.get_ResNet34( input_size, 1, "sigmoid", "avg", 
                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
                
            elif hyperparameters["architecture"].lower() == "custom_resnet50":
                model  = builder.get_ResNet50( input_size, 1, "sigmoid", "avg", 
                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
                
            elif hyperparameters["architecture"].lower() == "custom_resnet101":
                model  = builder.get_ResNet101( input_size, 1, "sigmoid", "avg", 
                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
                
            elif hyperparameters["architecture"].lower() == "custom_resnet152":
                model  = builder.get_ResNet152( input_size, 1, "sigmoid", "avg", 
                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "densenet" in hyperparameters["architecture"].lower():
            builder = DenseNet()
            
            if hyperparameters["architecture"].lower() == "custom_densenet121":
                model  = builder.get_DenseNet121( input_size, 1, "sigmoid", "avg", 
                                        hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                        hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif hyperparameters["architecture"].lower() == "custom_densenet169":
                model  = builder.get_DenseNet169( input_size, 1, "sigmoid", "avg", 
                                        hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                        hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif hyperparameters["architecture"].lower() == "custom_densenet201":
                model  = builder.get_DenseNet201( input_size, 1, "sigmoid", "avg", 
                                        hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                        hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif hyperparameters["architecture"].lower() == "custom_densenet264":
                model  = builder.get_DenseNet264( input_size, 1, "sigmoid", "avg", 
                                        hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                        hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "inception" in hyperparameters["architecture"].lower():
            builder = Inception()
            
            if hyperparameters["architecture"].lower() == "custom_inceptionv3":
                model = builder.get_InceptionV3( input_size, 1, "sigmoid", "avg", 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif hyperparameters["architecture"].lower() == "custom_inceptionv4":
                model = builder.get_InceptionV4( input_size, 1, "sigmoid", "avg", 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "xception" in hyperparameters["architecture"].lower():
            builder = Xception()
            
            model = builder.get_Xception( input_size, 1, "sigmoid", "avg", 
                                        hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                        hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "mobilenet" in hyperparameters["architecture"].lower():
            builder = MobileNet()
            alpha = float(hyperparameters["architecture"].split("_")[-1])
            architecture_name = "_".join(hyperparameters["architecture"].split("_")[:-1])
            print(f"Got Arq.: '{architecture_name}' with alpha: '{alpha}'")
            
            if architecture_name.lower() == "custom_mobilenetv2":
                model = builder.get_MobileNetV2( input_size, alpha, 6., 1, "sigmoid", "avg", 
                                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if architecture_name.lower() == "custom_mobilenetv3_small":
                model = builder.get_MobileNetV3_Small( input_size, alpha, 6., 1, "sigmoid", "avg", 
                                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if architecture_name.lower() == "custom_mobilenetv3_large":
                model = builder.get_MobileNetV3_Large( input_size, alpha, 6., 1, "sigmoid", "avg", 
                                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "efficientnet" in hyperparameters["architecture"].lower():
            builder = EfficientNet()
            is_v2 = "v2" in hyperparameters["architecture"].lower()
            
            if "_b0" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetB0( input_size, 1, "sigmoid", "avg", is_v2,
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if "_b1" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetB1( input_size, 1, "sigmoid", "avg", is_v2,
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if "_b2" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetB2( input_size, 1, "sigmoid", "avg", is_v2,
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if "_b3" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetB3( input_size, 1, "sigmoid", "avg", is_v2,
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if "_b4" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetB4( input_size, 1, "sigmoid", "avg", is_v2,
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if "_b5" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetB5( input_size, 1, "sigmoid", "avg", is_v2,
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if "_b6" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetB6( input_size, 1, "sigmoid", "avg", is_v2,
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if "_b7" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetB7( input_size, 1, "sigmoid", "avg", is_v2,
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if "v2_s" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetV2_S( input_size, 1, "sigmoid", "avg", 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if "v2_m" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetV2_M( input_size, 1, "sigmoid", "avg", 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if "v2_l" in hyperparameters["architecture"].lower():
                model = builder.get_EfficientNetV2_L( input_size, 1, "sigmoid", "avg", 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        else:
            raise ValueError(f"\nUnknown architecture '{hyperparameters['architecture'].lower()}'...")
            
        
        return model