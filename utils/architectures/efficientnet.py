import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D

DEFAULT_BLOCKS_ARGS = {
    "efficientnetv2-s": [{
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 24,
        "output_filters": 24,
        "expand_ratio": 1,
        "se_ratio": 0.0,
        "strides": 1,
        "conv_type": 1,
    }, {
        "kernel_size": 3,
        "num_repeat": 4,
        "input_filters": 24,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0.0,
        "strides": 2,
        "conv_type": 1,
    }, {
        "conv_type": 1,
        "expand_ratio": 4,
        "input_filters": 48,
        "kernel_size": 3,
        "num_repeat": 4,
        "output_filters": 64,
        "se_ratio": 0,
        "strides": 2,
    }, {
        "conv_type": 0,
        "expand_ratio": 4,
        "input_filters": 64,
        "kernel_size": 3,
        "num_repeat": 6,
        "output_filters": 128,
        "se_ratio": 0.25,
        "strides": 2,
    }, {
        "conv_type": 0,
        "expand_ratio": 6,
        "input_filters": 128,
        "kernel_size": 3,
        "num_repeat": 9,
        "output_filters": 160,
        "se_ratio": 0.25,
        "strides": 1,
    }, {
        "conv_type": 0,
        "expand_ratio": 6,
        "input_filters": 160,
        "kernel_size": 3,
        "num_repeat": 15,
        "output_filters": 256,
        "se_ratio": 0.25,
        "strides": 2,
    }],
    "efficientnetv2-m": [
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-l": [
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b0": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b1": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b2": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b3": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}

class EfficientNet:

    def __init__(self):
        self.args_v1 = [ # Args for EfficientNetV1 B0 ~ B7
                         { # Block 1 --------------------------------------------
                          "kernel_size": 3,
                          "repeats": 1,
                          "filters_in": 32,
                          "filters_out": 16,
                          "expand_ratio": 1,
                          "id_skip": True,
                          "strides": 1,
                          "se_ratio": 0.25 }, 
                         { # Blocks 2 and 3 ------------------------------------
                          "kernel_size": 3,
                          "repeats": 2,
                          "filters_in": 16,
                          "filters_out": 24,
                          "expand_ratio": 6,
                          "id_skip": True,
                          "strides": 2,
                          "se_ratio": 0.25 }, 
                         { # Blocks 4 and 5 ------------------------------------
                          "kernel_size": 5,
                          "repeats": 2,
                          "filters_in": 24,
                          "filters_out": 40,
                          "expand_ratio": 6,
                          "id_skip": True,
                          "strides": 2,
                          "se_ratio": 0.25 }, 
                         { # Blocks 6, 7 and 8 ---------------------------------
                          "kernel_size": 3,
                          "repeats": 3,
                          "filters_in": 40,
                          "filters_out": 80,
                          "expand_ratio": 6,
                          "id_skip": True,
                          "strides": 2,
                          "se_ratio": 0.25 }, 
                         { # Blocks 9, 10 and 11 -------------------------------
                          "kernel_size": 5,
                          "repeats": 3,
                          "filters_in": 80,
                          "filters_out": 112,
                          "expand_ratio": 6,
                          "id_skip": True,
                          "strides": 1,
                          "se_ratio": 0.25 }, 
                         { # Blocks 12, 13, 14 and 15 --------------------------
                          "kernel_size": 5,
                          "repeats": 4,
                          "filters_in": 112,
                          "filters_out": 192,
                          "expand_ratio": 6,
                          "id_skip": True,
                          "strides": 2,
                          "se_ratio": 0.25 }, 
                         { # Block 16 ------------------------------------------
                          "kernel_size": 3,
                          "repeats": 1,
                          "filters_in": 192,
                          "filters_out": 320,
                          "expand_ratio": 6,
                          "id_skip": True,
                          "strides": 1,
                          "se_ratio": 0.25
                  }]
        return
    
    def __call__( self, input_shape: tuple[int], w_coef: float, d_coef: float, model_args: dict, 
                  num_outputs: int, output_activation: str, pool: bool, base_dropout: float, 
                  top_dropout: float, l1_val: float, l2_val: float ) -> Model:

        # Model's input layer
        input_layer = tf.keras.layers.Input( shape = input_shape, name = "Input" )
        
        # Entry flow
        n_filters = EfficientNet.make_divisible(32 * w_coef, 8)
        x = EfficientNet.conv_bn( input_layer, num_filters = n_filters, kernel_size = 3, strides = 2, 
                                  padding = "same", activation = True, block = 1, num = 1, 
                                  dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val )
        
        # Middle flow
        block = 2
        for args in model_args:
            # Update block input and output filters based on depth multiplier.
            args["filters_in"] = EfficientNet.make_divisible(args["filters_in"] * w_coef, 8)
            args["filters_out"] = EfficientNet.make_divisible(args["filters_out"] * w_coef, 8)
            
            # Number of repetitions for the current block
            repeats = int(np.ceil(d_coef * args["repeats"]))
            
            for i in range(repeats):
                
                # Updates parameters that change from first repeat to the rest
                if i > 0:
                    args["strides"] = 1
                    args["filters_in"] = args["filters_out"]
                
                # Adds a block
                x = EfficientNet.inv_res_block( x, args["filters_in"], args["filters_out"], args["expand_ratio"], 
                                             args["kernel_size"], args["strides"], args["se_ratio"], args["id_skip"], 
                                             block = block, dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val )
                block += 1
            
        # Exit flow
        n_filters = EfficientNet.make_divisible(1280 * w_coef, 8)
        x = EfficientNet.conv_bn( x, num_filters = n_filters, kernel_size = 1, strides = 1, padding = "same", 
                                  activation = True, block = block, num = 1, dropchance = base_dropout, 
                                  l1_val = l1_val, l2_val = l2_val )
        
        # Global pooling used
        if pool.lower() == "max":
            x = GlobalMaxPooling2D(name = "GlobalMaxPooling")(x)
        else:
            x = GlobalAveragePooling2D(name = "GlobalAvgPooling")(x)

        # Adds Dropout before the output dense layer if top_dropout > 0
        if top_dropout > 0:
            x = tf.keras.layers.Dropout(top_dropout, name = f"topDropout_num1")(x)

        # Adds output layer
        output_layer = tf.keras.layers.Dense( num_outputs, activation = output_activation, name = "Classification_Layer" )(x)
        

        return tf.keras.models.Model( input_layer, output_layer )
    
    def get_EfficientNetB0(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                           base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 224 x 224 x 3 (but other input resolutions can be used)
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 1.0, d_coef = 1.0, model_args = self.args_v1,
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_EfficientNetB1(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                           base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 240 x 240 x 3 (but other input resolutions can be used)
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 1.0, d_coef = 1.1, model_args = self.args_v1,
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_EfficientNetB2(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                           base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 260 x 260 x 3 (but other input resolutions can be used)
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 1.1, d_coef = 1.2, model_args = self.args_v1,
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_EfficientNetB3(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                           base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 300 x 300 x 3 (but other input resolutions can be used)
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 1.2, d_coef = 1.4, model_args = self.args_v1,
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_EfficientNetB4(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                           base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 380 x 380 x 3 (but other input resolutions can be used)
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 1.4, d_coef = 1.8, model_args = self.args_v1,
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_EfficientNetB5(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                           base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 456 x 456 x 3 (but other input resolutions can be used)
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 1.6, d_coef = 2.2, model_args = self.args_v1,
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_EfficientNetB6(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                           base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 528 x 528 x 3 (but other input resolutions can be used)
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 1.8, d_coef = 2.6, model_args = self.args_v1,
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_EfficientNetB7(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                           base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 600 x 600 x 3 (but other input resolutions can be used)
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 2.0, d_coef = 3.1, model_args = self.args_v1,
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    @staticmethod
    def make_divisible(val, div):
        """ Rounds val to the nearest number divisible by div.
        If the closest divisible number is lower than 90% of val, 
        the new_val is rounded up instead. """
        new_val = max([div, div * (int(val + div / 2) // div) ])
        
        # Make sure that round down does not go down by more than 10%.
        if new_val < 0.9 * val:
            new_val += div
        return new_val
    
    @staticmethod
    def squeeze_excite_block( layer_input: tf.Tensor, filters: int, ratio: float, block: int, num: int ) -> tf.Tensor:
        
        # Pooling
        x = GlobalAveragePooling2D( keepdims = True, name = f"se_block{block}_GAvgPool_num{num}")(layer_input)
        expanded_filters = backend.int_shape(x)[-1]
        
        # First Conv
        n_filters = max(1, int(filters * ratio))
        x = Conv2D( filters = n_filters, kernel_size = 1, strides = 1, padding = "same", 
                    use_bias = True, kernel_initializer = "he_uniform", activation = "swish",
                    name = f"se_block{block}_Conv2D_num{num}" )(x)
        
        # Second Conv
        x = Conv2D( filters = expanded_filters, kernel_size = 1, strides = 1, padding = "same", 
                    use_bias = True, kernel_initializer = "he_uniform", activation = "sigmoid",
                    name = f"se_block{block}_Conv2D_num{num+1}" )(x)
        
        # Multiplication
        out = Multiply(name = f"se_block{block}_Mult_num{num}")([x, layer_input])
        
        return out
    
    @staticmethod
    def conv_bn( x: tf.Tensor, num_filters: int, kernel_size: int, strides: int, padding: str, activation: bool, 
                 block: int, num: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        if dropchance > 0:
            x = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num{num}")(x)
        
        x = Conv2D( filters = num_filters, kernel_size = kernel_size, strides = strides, padding = padding,
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num{num}" )(x)
        
        x = BatchNormalization(name = f"block{block}_BN_num{num}")(x)
        
        # Uses Swish activation
        if activation:
            x = Activation("swish", name = f"block{block}_Swish_num{num}")(x)
        
        return x
    
    @staticmethod
    def dw_conv_bn_relu( x: tf.Tensor, kernel_size: int, strides: int, padding: str, activation: bool, 
                         block: int, num: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        if dropchance > 0:
            x = SpatialDropout2D(dropchance, name = f"block{block}_dwSDropout_num{num}")(x)
        
        x = DepthwiseConv2D( kernel_size = kernel_size, strides = strides, padding = padding, use_bias = False,
                             depthwise_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                             depthwise_initializer = "he_uniform", name = f"block{block}_dwConv2D_num{num}" )(x)
        
        x = BatchNormalization(name = f"block{block}_dwBN_num{num}")(x)
        
        # Uses Swish activation
        if activation:
            x = Activation("swish", name = f"dw_block{block}_Swish_num{num}")(x)
        
        return x
    
    @staticmethod
    def inv_res_block( x: tf.Tensor, filters_in: int, filters_out: int, expansion: float, 
                       kernel_size: int, strides: int, se_ratio: float, id_skip: bool, block: int, 
                       dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        expansion_filters = expansion * filters_in
        
        num = 1
        if expansion != 1:
            y = EfficientNet.conv_bn( x, num_filters = expansion_filters, kernel_size = 1, strides = 1, padding = "same", 
                                      activation = True, block = block, num = num, dropchance = dropchance, 
                                      l1_val = l1_val, l2_val = l2_val)
            num += 1
            
        y = EfficientNet.dw_conv_bn_relu( y if (expansion != 1) else x, kernel_size = kernel_size, strides = strides, padding = "same", 
                                       activation = True, block = block, num = 1, dropchance = dropchance, 
                                       l1_val = l1_val, l2_val = l2_val)
        
        if not se_ratio is None:
            y = EfficientNet.squeeze_excite_block( layer_input = y, filters = filters_in, ratio = se_ratio, 
                                                   block = block, num = 1 )
        
        y = EfficientNet.conv_bn( y, num_filters = filters_out, kernel_size = 1, strides = 1, padding = "same", 
                                  activation = False, block = block, num = num, dropchance = dropchance, 
                                  l1_val = l1_val, l2_val = l2_val)
        
        in_channels = backend.int_shape(x)[-1]
        if id_skip and in_channels == filters_out and strides == 1:
            y = Add(name = f"block{block}_Add_num1")([x, y])
        return y