import copy
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

class EfficientNet:

    def __init__(self):
        self.args_v1 = [ # Args for EfficientNetV1 B0 ~ B7
                         { # Block 1 -----------------------------------------
                          "kernel_size": 3,
                          "num_repeat": 1,
                          "input_filters": 32,
                          "output_filters": 16,
                          "expand_ratio": 1,
                          "se_ratio": 0.25,
                          "strides": 1,
                          "conv_type": 0 }, 
                         { # Blocks 2 and 3 ----------------------------------
                          "kernel_size": 3,
                          "num_repeat": 2,
                          "input_filters": 16,
                          "output_filters": 24,
                          "expand_ratio": 6,
                          "se_ratio": 0.25,
                          "strides": 2,
                          "conv_type": 0 }, 
                         { # Blocks 4 and 5 ----------------------------------
                          "kernel_size": 5,
                          "num_repeat": 2,
                          "input_filters": 24,
                          "output_filters": 40,
                          "expand_ratio": 6,
                          "se_ratio": 0.25,
                          "strides": 2,
                          "conv_type": 0 }, 
                         { # Blocks 6, 7 and 8 -------------------------------
                          "kernel_size": 3,
                          "num_repeat": 3,
                          "input_filters": 40,
                          "output_filters": 80,
                          "expand_ratio": 6,
                          "se_ratio": 0.25,
                          "strides": 2,
                          "conv_type": 0 }, 
                         { # Blocks 9, 10 and 11 -----------------------------
                          "kernel_size": 5,
                          "num_repeat": 3,
                          "input_filters": 80,
                          "output_filters": 112,
                          "expand_ratio": 6,
                          "se_ratio": 0.25,
                          "strides": 1,
                          "conv_type": 0 }, 
                         { # Blocks 12, 13, 14 and 15 ------------------------
                          "kernel_size": 5,
                          "num_repeat": 4,
                          "input_filters": 112,
                          "output_filters": 192,
                          "expand_ratio": 6,
                          "se_ratio": 0.25,
                          "strides": 2,
                          "conv_type": 0 }, 
                         { # Block 16 ----------------------------------------
                          "kernel_size": 3,
                          "num_repeat": 1,
                          "input_filters": 192,
                          "output_filters": 320,
                          "expand_ratio": 6,
                          "se_ratio": 0.25,
                          "strides": 1,
                          "conv_type": 0 }
                       ]
        
        self.args_v2 = [ # Args for EfficientNetV2 B0 ~ B3
                         { # Block 1 -----------------------------------------
                          "kernel_size": 3,
                          "num_repeat": 1,
                          "input_filters": 32,
                          "output_filters": 16,
                          "expand_ratio": 1,
                          "se_ratio": 0,
                          "strides": 1,
                          "conv_type": 1, },
                         { # Blocks 2 and 3 ----------------------------------
                          "kernel_size": 3,
                          "num_repeat": 2,
                          "input_filters": 16,
                          "output_filters": 32,
                          "expand_ratio": 4,
                          "se_ratio": 0,
                          "strides": 2,
                          "conv_type": 1, },
                         { # Blocks 4 and 5 ----------------------------------
                          "kernel_size": 3,
                          "num_repeat": 2,
                          "input_filters": 32,
                          "output_filters": 48,
                          "expand_ratio": 4,
                          "se_ratio": 0,
                          "strides": 2,
                          "conv_type": 1, },
                         { # Blocks 6, 7 and 8 -------------------------------
                          "kernel_size": 3,
                          "num_repeat": 3,
                          "input_filters": 48,
                          "output_filters": 96,
                          "expand_ratio": 4,
                          "se_ratio": 0.25,
                          "strides": 2,
                          "conv_type": 0, },
                         { # Blocks 9 - 13 -----------------------------------
                          "kernel_size": 3,
                          "num_repeat": 5,
                          "input_filters": 96,
                          "output_filters": 112,
                          "expand_ratio": 6,
                          "se_ratio": 0.25,
                          "strides": 1,
                          "conv_type": 0, },
                         { # Blocks 14 - 23 ----------------------------------
                          "kernel_size": 3,
                          "num_repeat": 8,
                          "input_filters": 112,
                          "output_filters": 192,
                          "expand_ratio": 6,
                          "se_ratio": 0.25,
                          "strides": 2,
                          "conv_type": 0, },
                     ]
        
        return
    
    def __call__( self, input_shape: tuple[int], w_coef: float, d_coef: float, model_args: dict, 
                  v2: bool, num_outputs: int, output_activation: str, pool: bool, base_dropout: float, 
                  top_dropout: float, l1_val: float, l2_val: float ) -> Model:

        # Model's input layer
        input_layer = tf.keras.layers.Input( shape = input_shape, name = "Input" )
        
        # Entry flow
        base_filters = model_args[0]["input_filters"]
        n_filters = EfficientNet.make_divisible( base_filters * w_coef, 8, check_round = (not v2) )
        x = EfficientNet.conv_bn( input_layer, num_filters = n_filters, kernel_size = 3, strides = 2, 
                                  padding = "same", activation = True, block = 1, num = 1, 
                                  dropchance = 0, l1_val = l1_val, l2_val = l2_val )
        
        # Middle flow
        block = 2
        for args in model_args:
            # Update block input and output filters based on depth multiplier.
            args["input_filters"] = EfficientNet.make_divisible(args["input_filters"] * w_coef, 8, check_round = (not v2) )
            args["output_filters"] = EfficientNet.make_divisible(args["output_filters"] * w_coef, 8, check_round = (not v2) )
            
            # Number of repetitions for the current block
            repeats = int(np.ceil(d_coef * args["num_repeat"]))
            
            for i in range(repeats):
                
                # Updates parameters that change from first repeat to the rest
                if i > 0:
                    args["strides"] = 1
                    args["input_filters"] = args["output_filters"]
                
                if args["conv_type"] == 0:
                    # Adds an Inverse Residual Block
                    x = EfficientNet.inv_res_block( x, args["input_filters"], args["output_filters"], args["expand_ratio"], 
                                                    args["kernel_size"], args["strides"], args["se_ratio"], block = block, 
                                                    dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val )
                
                else:
                    # Adds a Fused Inverse Residual Block (Expansion Conv and DepthWise Conv are combined)
                    x = EfficientNet.fused_inv_res_block( x, args["input_filters"], args["output_filters"], args["expand_ratio"], 
                                                          args["kernel_size"], args["strides"], args["se_ratio"], block = block, 
                                                          dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val )
                block += 1
            
        # Exit flow
        n_filters = EfficientNet.make_divisible(1280 * w_coef, 8, check_round = (not v2) )
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
    
    def get_EfficientNetB(self, input_shape: tuple[int], num_outputs: int, 
                          output_activation: str, pool: bool, v2: bool, 
                          b: int, base_dropout: float, top_dropout: float, 
                          l1_val: float, l2_val: float) -> Model:

        # Dict with w_coef and d_coef based on B models.
        # Reference input shape for each model is given. 
        # Howerver, other resolutions can also be used.
        param_dict = { 0: (1.0, 1.0), # 224 x 224 x 3
                       1: (1.0, 1.1), # 240 x 240 x 3
                       2: (1.1, 1.2), # 260 x 260 x 3
                       3: (1.2, 1.4), # 300 x 300 x 3
                       4: (1.4, 1.8), # 380 x 380 x 3
                       5: (1.6, 2.2), # 456 x 456 x 3
                       6: (1.8, 2.6), # 528 x 528 x 3
                       7: (2.0, 3.1)  # 600 x 600 x 3
                     }
        w_coef, d_coef = param_dict[b]
        
        # Uses the base params accordingly to the selected version
        model_args = self.args_v2 if v2 else self.args_v1
        
        # Uses the call function to build the model
        model = self( input_shape, w_coef, d_coef, model_args, v2, num_outputs,
                      output_activation, pool, base_dropout, top_dropout,
                      l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_EfficientNetV2_S(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                             base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 384 x 384 x 3 (but other input resolutions can be used)
        
        # Adapts EfficientNetV2_B models to EfficientNetV2_S
        model_args = copy.deepcopy(self.args_v2)
        
        # List of "num_repeat" property for EfficientNetV2_S model blocks
        repeat_list = [ 2, 4, 4, 6, 9, 15 ]
        
        # List of input/output filters for EfficientNetV2_S model blocks
        filter_list = [ 24, 24, 48, 64, 128, 160, 256 ]
        
        # Adjusts values 
        for i, repeats in enumerate(repeat_list):
            model_args[i]["num_repeat"] = repeats
            model_args[i]["input_filters"] = filter_list[i]
            model_args[i]["output_filters"] = filter_list[i+1]
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 1.0, d_coef = 1.0, model_args = model_args, v2 = True, 
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_EfficientNetV2_M(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                             base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 480 x 480 x 3 (but other input resolutions can be used)
        
        # Adapts EfficientNetV2_B models to EfficientNetV2_M
        model_args = copy.deepcopy(self.args_v2)
        
        # Adds an extra block at the end of the list in model_args
        extra_block = { "kernel_size": 3,
                        "num_repeat": 5,
                        "input_filters": 304,
                        "output_filters": 512,
                        "expand_ratio": 6,
                        "se_ratio": 0.25,
                        "strides": 1,
                        "conv_type": 0 
                      } 
        model_args.append( extra_block )
        
        # List of "num_repeat" property for EfficientNetV2_M model blocks
        repeat_list = [ 3, 5, 5, 7, 14, 18, 5 ]
        
        # List of input/output filters for EfficientNetV2_M model blocks
        filter_list = [ 24, 24, 48, 80, 160, 176, 304, 512 ]
        
        # Adjusts values 
        for i, repeats in enumerate(repeat_list):
            model_args[i]["num_repeat"] = repeats
            model_args[i]["input_filters"] = filter_list[i]
            model_args[i]["output_filters"] = filter_list[i+1]
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 1.0, d_coef = 1.0, model_args = model_args, v2 = True, 
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_EfficientNetV2_L(self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool, 
                             base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        # Designed for 480 x 480 x 3 (but other input resolutions can be used)
        
        # Adapts EfficientNetV2_B models to EfficientNetV2_L
        model_args = copy.deepcopy(self.args_v2)
        
        # Adds an extra block at the end of the list in model_args
        extra_block = { "kernel_size": 3,
                        "num_repeat": 7,
                        "input_filters": 384,
                        "output_filters": 640,
                        "expand_ratio": 6,
                        "se_ratio": 0.25,
                        "strides": 1,
                        "conv_type": 0 
                      } 
        model_args.append( extra_block )
        
        # List of "num_repeat" property for EfficientNetV2_L model blocks
        repeat_list = [ 4, 7, 7, 10, 19, 25, 7 ]
        
        # List of input/output filters for EfficientNetV2_L model blocks
        filter_list = [ 32, 32, 64, 96, 192, 224, 384, 640 ]
        
        # Adjusts values 
        for i, repeats in enumerate(repeat_list):
            model_args[i]["num_repeat"] = repeats
            model_args[i]["input_filters"] = filter_list[i]
            model_args[i]["output_filters"] = filter_list[i+1]
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, w_coef = 1.0, d_coef = 1.0, model_args = model_args, v2 = True, 
                     num_outputs = num_outputs, output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    @staticmethod
    def make_divisible(val, div, check_round = True):
        """ Rounds val to the nearest number divisible by div.
        If the closest divisible number is lower than 90% of val, 
        the new_val is rounded up instead. """
        new_val = max([div, div * (int(val + div / 2) // div) ])
        
        # Checks if round down goes down by more than 10%.
        # If specified, rounds up for losses above 10%.
        if check_round and (new_val < 0.9 * val):
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
                       kernel_size: int, strides: int, se_ratio: float, block: int, 
                       dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        y = x
        expansion_filters = expansion * filters_in
        
        num = 1
        if expansion != 1:
            y = EfficientNet.conv_bn( y, num_filters = expansion_filters, kernel_size = 1, strides = 1, padding = "same", 
                                      activation = True, block = block, num = num, dropchance = 0, 
                                      l1_val = l1_val, l2_val = l2_val)
            num += 1
            
        y = EfficientNet.dw_conv_bn_relu( y, kernel_size = kernel_size, strides = strides, padding = "same", 
                                          activation = True, block = block, num = 1, dropchance = dropchance, 
                                          l1_val = l1_val, l2_val = l2_val)
        
        if (not se_ratio is None) and (se_ratio > 0):
            y = EfficientNet.squeeze_excite_block( layer_input = y, filters = filters_in, ratio = se_ratio, 
                                                   block = block, num = 1 )
        
        y = EfficientNet.conv_bn( y, num_filters = filters_out, kernel_size = 1, strides = 1, padding = "same", 
                                  activation = False, block = block, num = num, dropchance = dropchance, 
                                  l1_val = l1_val, l2_val = l2_val)
        
        in_channels = backend.int_shape(x)[-1]
        if in_channels == filters_out and strides == 1:
            y = Add(name = f"block{block}_Add_num1")([x, y])
        return y
    
    @staticmethod
    def fused_inv_res_block( x: tf.Tensor, filters_in: int, filters_out: int, expansion: float, 
                             kernel_size: int, strides: int, se_ratio: float, block: int, 
                             dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        y = x
        expansion_filters = expansion * filters_in
        
        num = 1
        if expansion != 1:
            y = EfficientNet.conv_bn( y, num_filters = expansion_filters, kernel_size = kernel_size, 
                                      strides = strides, padding = "same", activation = True,
                                      block = block, num = num, dropchance = 0, 
                                      l1_val = l1_val, l2_val = l2_val)
            num += 1
        
        if (not se_ratio is None) and (se_ratio > 0):
            y = EfficientNet.squeeze_excite_block( layer_input = y, filters = filters_in, ratio = se_ratio, 
                                                   block = block, num = 1 )
        
        kernel_size = 1 if expansion != 1 else kernel_size
        y = EfficientNet.conv_bn( y, num_filters = filters_out, kernel_size = kernel_size, strides = 1, 
                                  padding = "same", activation = False, block = block, num = num, 
                                  dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        in_channels = backend.int_shape(x)[-1]
        if in_channels == filters_out and strides == 1:
            y = Add(name = f"block{block}_Add_num1")([x, y])
        return y