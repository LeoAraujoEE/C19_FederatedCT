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
        model = self.create_model(hyperparameters)
        
        # Counts the model's parameters
        trainable_count, non_trainable_count = self.count_model_params(model)
        print("\nCreated model with {:,} trainable parameters and {:,} non trainable ones...".format(trainable_count, non_trainable_count))

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
        # Input dimensions
        input_size = (hyperparameters["input_height"], 
                      hyperparameters["input_width"], 
                      hyperparameters["input_channels"])
        
        # Architecture name
        arch_name = hyperparameters["architecture"].lower()
        
        if "inception_resnet" in arch_name:
            builder = InceptionResNet()
            model  = builder.get_InceptionResNetV2( input_size, 1, "sigmoid", "avg", 
                             hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                             hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "resnet" in arch_name:
            builder = ResNet()
            
            if arch_name == "resnet18":
                model  = builder.get_ResNet18( input_size, 1, "sigmoid", "avg", 
                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
                
            elif arch_name == "resnet34":
                model  = builder.get_ResNet34( input_size, 1, "sigmoid", "avg", 
                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
                
            elif arch_name == "resnet50":
                model  = builder.get_ResNet50( input_size, 1, "sigmoid", "avg", 
                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
                
            elif arch_name == "resnet101":
                model  = builder.get_ResNet101( input_size, 1, "sigmoid", "avg", 
                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
                
            elif arch_name == "resnet152":
                model  = builder.get_ResNet152( input_size, 1, "sigmoid", "avg", 
                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "densenet" in arch_name:
            builder = DenseNet()
            
            if arch_name == "densenet121":
                model  = builder.get_DenseNet121( input_size, 1, "sigmoid", "avg", 
                                        hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                        hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif arch_name == "densenet169":
                model  = builder.get_DenseNet169( input_size, 1, "sigmoid", "avg", 
                                        hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                        hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif arch_name == "densenet201":
                model  = builder.get_DenseNet201( input_size, 1, "sigmoid", "avg", 
                                        hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                        hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif arch_name == "densenet264":
                model  = builder.get_DenseNet264( input_size, 1, "sigmoid", "avg", 
                                        hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                        hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "inception" in arch_name:
            builder = Inception()
            
            if arch_name == "inceptionv3":
                model = builder.get_InceptionV3( input_size, 1, "sigmoid", "avg", 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif arch_name == "inceptionv4":
                model = builder.get_InceptionV4( input_size, 1, "sigmoid", "avg", 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "xception" in arch_name:
            builder = Xception()
            
            model = builder.get_Xception( input_size, 1, "sigmoid", "avg", 
                                        hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                        hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
        
        elif "mobilenet" in arch_name:
            builder = MobileNet()
            # For the default names, alpha is considered to be 1.0
            if (arch_name in ["mobilenetv2", "mobilenetv3_small", 
                              "mobilenetv3_large"]):
                alpha = 1.0
                architecture_name = arch_name
            
            # Otherwise, extracts alpha from the architecture name
            else:
                alpha = float(hyperparameters["architecture"].split("_")[-1])
                architecture_name = "_".join(hyperparameters["architecture"].split("_")[:-1])
            
            if architecture_name.lower() == "mobilenetv2":
                model = builder.get_MobileNetV2( input_size, alpha, 6., 1, "sigmoid", "avg", 
                                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if architecture_name.lower() == "mobilenetv3_small":
                model = builder.get_MobileNetV3_Small( input_size, alpha, 6., 1, "sigmoid", "avg", 
                                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            if architecture_name.lower() == "mobilenetv3_large":
                model = builder.get_MobileNetV3_Large( input_size, alpha, 6., 1, "sigmoid", "avg", 
                                                hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                                hyperparameters["l1_reg"], hyperparameters["l2_reg"] )      

        elif ("efficientnet" in arch_name):
            is_v2 = "v2" in arch_name
            builder = EfficientNet()

            if "b" in arch_name:
                variant = int(arch_name.split("b")[-1])
                model = builder.get_EfficientNetB(input_size, 1, "sigmoid", "avg", is_v2, variant, 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif "s" in arch_name:
                model = builder.get_EfficientNetV2_S( input_size, 1, "sigmoid", "avg", 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif "m" in arch_name:
                model = builder.get_EfficientNetV2_M( input_size, 1, "sigmoid", "avg", 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] )
            
            elif "l" in arch_name:
                model = builder.get_EfficientNetV2_L( input_size, 1, "sigmoid", "avg", 
                                            hyperparameters["base_dropout"], hyperparameters["top_dropout"], 
                                            hyperparameters["l1_reg"], hyperparameters["l2_reg"] ) 
        else:
            raise ValueError(f"\nUnknown architecture '{hyperparameters['architecture'].lower()}'...")
            
        
        return model