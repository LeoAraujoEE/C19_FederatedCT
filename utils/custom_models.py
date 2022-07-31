import os
import json
import random
import tempfile
import numpy as np
import tensorflow as tf

from utils.architectures.resnet import ResNet
from utils.architectures.densenet import DenseNet
from utils.architectures.xception import Xception
from utils.architectures.inception import Inception
from utils.architectures.inception_resnet import InceptionResNet
from tensorflow.python.keras.utils.layer_utils import count_params

from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

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
            Resnets        :  18,  34,  50, 101, 152
            Densenets      : 121, 169, 201, 264
            Inception      :  V3,  V4
            InceptionResNet:  V2
        
        Old Supported:
            vgg_16, vgg_19, resnet_50v2, resnet_101v2, resnet_152v2, 
            mobilenet_v2, xception, densenet_121, densenet_169, densenet_201, 
            efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, 
            efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
        """
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Creates a model with all layers up to the end of the convolutional base
        if "custom" in hyperparameters["architecture"].lower():
            model  = self.create_model(hyperparameters)
            
        else:
            model = self.get_architecture( hyperparameters )

            # Adds L1 regularization if needed
            if hyperparameters["l1_reg"] > 0:
                regularizer = tf.keras.regularizers.l1( hyperparameters["l1_reg"] )
                model = self.add_regularization( model, regularizer = regularizer )

            # Adds L2 regularization if needed
            if hyperparameters["l2_reg"] > 0:
                regularizer = tf.keras.regularizers.l2( hyperparameters["l2_reg"] )
                model = self.add_regularization( model, regularizer = regularizer )
        
        # Counts the model's parameters
        trainable_count = int(np.sum([ count_params(l.trainable_weights) for l in model.layers ]))
        non_trainable_count = int(np.sum([ count_params(l.non_trainable_weights) for l in model.layers ]))
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
            
        
        return model

    def get_architecture(self, hyperparameters):
        inputH = hyperparameters["input_height"]
        inputW = hyperparameters["input_width"]
        inputC = hyperparameters["input_channels"]
        input_size = (inputH, inputW, inputC)

        if hyperparameters["architecture"].lower() == "xception":
            # Blocks 1 to 14
            # Recommended size -> 299 x 299
            base_model = tf.keras.applications.xception.Xception( input_shape = input_size, include_top = False, 
                                                            weights = None, pooling = "avg" )

        elif hyperparameters["architecture"].lower() in ["resnet_50v2", "resnet50v2"]:
            # Recommended size -> 224 x 224
            base_model = tf.keras.applications.ResNet50V2( input_shape = input_size, include_top = False, 
                                                     weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "resnet" )

        elif hyperparameters["architecture"].lower() in ["resnet_101v2", "resnet101v2"]:
            # Recommended size -> 224 x 224
            base_model = tf.keras.applications.ResNet101V2( input_shape = input_size, include_top = False, 
                                                      weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "resnet" )

        elif hyperparameters["architecture"].lower() in ["resnet_152v2", "resnet152v2"]:
            # Recommended size -> 224 x 224
            base_model = tf.keras.applications.ResNet152V2( input_shape = input_size, include_top = False, 
                                                      weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "resnet" )

        elif hyperparameters["architecture"].lower() in ["mobilenet_v2", "mobilenetv2"]:
            # Recommended size -> 224 x 224
            base_model = tf.keras.applications.MobileNetV2( input_shape = input_size, include_top = False, 
                                                      weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "mobilenet" )

        elif hyperparameters["architecture"].lower() in ["densenet_121", "densenet121"]:
            # Recommended size -> 224 x 224
            base_model = tf.keras.applications.DenseNet121( input_shape = input_size, include_top = False, 
                                                      weights = None, pooling = "avg" )

        elif hyperparameters["architecture"].lower() in ["densenet_169", "densenet169"]:
            # Recommended size -> 224 x 224
            base_model = tf.keras.applications.DenseNet169( input_shape = input_size, include_top = False, 
                                                      weights = None, pooling = "avg" )

        elif hyperparameters["architecture"].lower() in ["densenet_201", "densenet201"]:
            # Recommended size -> 224 x 224
            base_model = tf.keras.applications.DenseNet201( input_shape = input_size, include_top = False, 
                                                      weights = None, pooling = "avg")

        elif hyperparameters["architecture"].lower() in ["vgg_16", "vgg16"]:
            # Recommended size -> 224 x 224
            base_model = tf.keras.applications.VGG16( input_shape = input_size, include_top = False, 
                                                weights = None, pooling = "avg" )

        elif hyperparameters["architecture"].lower() in ["vgg_19", "vgg19"]:
            # Recommended size -> 224 x 224
            base_model = tf.keras.applications.VGG19( input_shape = input_size, include_top = False, 
                                                weights = None, pooling = "avg" )

        elif hyperparameters["architecture"].lower() in ["efficientnet_b0", "efficientnetb0"]:
            # Recommended size -> 224 x 224
            base_model = tf.keras.applications.EfficientNetB0( input_shape = input_size, include_top = False, 
                                                         weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "efficientnet" )

        elif hyperparameters["architecture"].lower() in ["efficientnet_b1", "efficientnetb1"]:
            # Recommended size -> 240 x 240
            base_model = tf.keras.applications.EfficientNetB1( input_shape = input_size, include_top = False, 
                                                         weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "efficientnet" )

        elif hyperparameters["architecture"].lower() in ["efficientnet_b2", "efficientnetb2"]:
            # Recommended size -> 260 x 260
            base_model = tf.keras.applications.EfficientNetB2( input_shape = input_size, include_top = False, 
                                                         weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "efficientnet" )

        elif hyperparameters["architecture"].lower() in ["efficientnet_b3", "efficientnetb3"]:
            # Recommended size -> 300 x 300
            base_model = tf.keras.applications.EfficientNetB3( input_shape = input_size, include_top = False, 
                                                         weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "efficientnet" )

        elif hyperparameters["architecture"].lower() in ["efficientnet_b4", "efficientnetsb4"]:
            # Recommended size -> 380 x 380
            base_model = tf.keras.applications.EfficientNetB4( input_shape = input_size, include_top = False, 
                                                         weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "efficientnet" )

        elif hyperparameters["architecture"].lower() in ["efficientnet_b5", "efficientnetsb5"]:
            # Recommended size -> 456 x 456
            base_model = tf.keras.applications.EfficientNetB5( input_shape = input_size, include_top = False, 
                                                         weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "efficientnet" )

        elif hyperparameters["architecture"].lower() in ["efficientnet_b6", "efficientnetsb6"]:
            # Recommended size -> 528 x 528
            base_model = tf.keras.applications.EfficientNetB6( input_shape = input_size, include_top = False, 
                                                         weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "efficientnet" )

        elif hyperparameters["architecture"].lower() in ["efficientnet_b7", "efficientnetsb7"]:
            # Recommended size -> 600 x 600
            base_model = tf.keras.applications.EfficientNetB7( input_shape = input_size, include_top = False, 
                                                         weights = None, pooling = "avg" )
            base_model = self.edit_layer_names( base_model, model_type = "efficientnet" )
        else:
            raise ValueError("\nUnknown architecture...")

        x = tf.keras.layers.Dropout( hyperparameters["top_dropout"], name = "topDropout_1" )(base_model.layers[-1].output)

        # Adds output layer
        output_layer = tf.keras.layers.Dense( 1, activation = "sigmoid", name = "Classification_Layer" )(x)

        return tf.keras.models.Model( base_model.input, output_layer )

    def unfreeze_blocks( self, model, n_blocks, print_bool = True ):

        max_block = self.get_max_block(model)

        # Starts conv_bool as False to keep
        # initial layers frozen
        conv_bool = False

        # Defines the 1st block to unfreeze
        # all subsequent blocks are also unfrozen
        targeted_block = "block{}_".format( max_block - n_blocks + 1 )

        # Iterates through the model's layers
        for layer in model.layers:
            # If we get to layers of the targenet block
            if targeted_block in layer.name:
                # Flips conv_bool to True
                conv_bool = True
            
            # Sets trainable parameter to conv_bool except for BN layers
            # if layer.__class__.__name__ == "BatchNormalization":  
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = conv_bool
        
        # Counts the model's parameters
        if print_bool:
            trainable_count = int(np.sum([ count_params(l.trainable_weights) for l in model.layers ]))
            non_trainable_count = int(np.sum([ count_params(l.non_trainable_weights) for l in model.layers ]))
            print("\tModel has {:,} trainable parameters and {:,} non trainable ones after unfreezing {}/{} blocks...".format(trainable_count, 
                                                                                                                        non_trainable_count, 
                                                                                                                        n_blocks, max_block))

        return model
    
    def get_max_block(self, model):
        blocks = [ self.get_block_num_from_layer_name(l.name) for l in model.layers if "block" in l.name ]
        max_block = np.max(blocks)
        return max_block

    def edit_layer_names( self, model, model_type, custom_objects = None ):
        '''Adjusts the block information on layer names for pre-trained models
        Based on: https://nrasadi.medium.com/change-model-layer-name-in-tensorflow-keras-58771dd6bf1b
        Arguments:
            model: a tf.keras model
            model_type: a string that specifies the type of model (i.e. resnet, inception, mobilenet)
            custom_objects: if your model consists of custom layers you shoud add them pass them as a dictionary. 
                For more information read the following:
                https://keras.io/guides/serialization_and_saving/#custom-objects
        Returns:
            new_model: a tf.keras model having same weights as the input model.
        '''
        
        config = model.get_config()
        old_to_new = {}
        new_to_old = {}
        
        current_block = 1
        for layer in config["layers"]:

            # Sets the correct block number based on current_block
            new_block_num = "_block{}_".format( current_block )

            if "block" in layer["name"].lower():
                # Add/Concat layers mark the end of a block
                if layer["class_name"] in ["Add", "Concatenate"]:
                    # Hence, if such layer is found, the current block number increases
                    current_block += 1

                # Edits the layer name to set the correct block number
                if model_type == "resnet":
                    # For resnets the block info is inconsistent and needs to be replaced
                    old_block_num = "_{}_".format( layer["name"].split("_")[1] )
                    new_name = "n_"+layer["name"].replace( old_block_num, new_block_num )

                elif model_type == "mobilenet":
                    # For mobilenets the block information is correct, but has an extra "_"
                    new_name = "n_"+layer["name"].replace( "block_", "block" )
                
                elif model_type == "efficientnet":
                    # For efficientnets the block information is correct, but has an extra "a", "b" or "c"
                    old_block_num = "block{}".format( self.get_block_num_from_layer_name( layer["name"] ) )
                    new_name = "n_"+layer["name"].replace( old_block_num, old_block_num+"_" )
                
                else:
                    raise ValueError("\nUnknown model type specified...")

            else:
                new_name = "n_"+layer["name"]

            # Uses dicts to associate the old names to the new ones
            old_to_new[layer["name"]], new_to_old[new_name] = new_name, layer["name"]
            
            # Changes the layer's name in the model's config and in the own layer's configs
            layer["name"] = new_name
            layer["config"]["name"] = new_name

            # Also applies changes to inbound nodes
            if len(layer["inbound_nodes"]) > 0:
                for in_node in layer["inbound_nodes"][0]:
                    in_node[0] = old_to_new[in_node[0]]
        
        # Applies changes to all input layers across the config file
        for input_layer in config["input_layers"]:
            input_layer[0] = old_to_new[input_layer[0]]
        
        # Applies changes to all output layers across the config file
        for output_layer in config["output_layers"]:
            output_layer[0] = old_to_new[output_layer[0]]
        
        # Changes the model name to avoid different models with the same name
        config["name"] = "updated_" + config["name"]
        new_model = tf.keras.Model().from_config( config, custom_objects )
        
        # Copies the weights from original model to the new one
        for layer in new_model.layers:
            layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())
        
        return new_model
    
    @staticmethod
    def get_block_num_from_layer_name( l_name ):
        block_num_str = l_name.split("block")[-1].split("_")[0]
        block_num_int = int( "".join(filter(str.isdigit, block_num_str)) )
        return block_num_int

    @staticmethod
    def add_regularization( model, regularizer ):
        """ Adds L1 or L2 regularization to layers from a pretrained model """
        # source: https://towardsdatascience.com/how-to-add-regularization-to-keras-pre-trained-models-the-right-way-743776451091#:~:text=You%20can%20pass%20any%20model,returns%20the%20model%20properly%20configured.

        if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
            print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
            return model

        for layer in model.layers:
            for attr in ["kernel_regularizer"]:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        # When we change the layers attributes, the change only happens in the model config file
        model_json = model.to_json()

        # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(tempfile.gettempdir(), "tmp_weights.h5")
        model.save_weights(tmp_weights_path)

        # load the model from the config
        model = tf.keras.models.model_from_json(model_json)

        # Reload the model weights
        model.load_weights(tmp_weights_path, by_name=True)
        return model