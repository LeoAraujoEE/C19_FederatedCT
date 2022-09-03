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

class MobileNet:

    def __init__(self):
        return
    
    def __call__( self, input_shape: tuple[int], alpha: float, expansion: int, num_outputs: int, 
                  output_activation: str, pool: bool, base_dropout: float, top_dropout: float, 
                  l1_val: float, l2_val: float, v3: bool=False, small: bool=False ) -> Model:

        # Model's input layer
        input_layer = tf.keras.layers.Input( shape = input_shape, name = "Input" )
        
        # Entry Flow:
        x = MobileNet.entry_flow(input_layer, alpha = alpha, v3 = v3, 
                                 dropchance = 0, l1_val = l1_val, 
                                 l2_val = l2_val )
        
        if v3:
            if small:
                x = MobileNet.middle_flow_v3_small( x, alpha, expansion, dropchance = base_dropout, 
                                                    l1_val = l1_val, l2_val = l2_val )
            else:
                x = MobileNet.middle_flow_v3_large( x, alpha, expansion, dropchance = base_dropout, 
                                                    l1_val = l1_val, l2_val = l2_val )
                
            output_layer = MobileNet.exit_flow_v3( x, alpha = alpha, expansion = expansion, small = small, num_outputs = num_outputs, 
                                                  output_activation = output_activation, pool = pool, base_dropout = base_dropout, 
                                                  top_dropout = top_dropout, l1_val = l1_val, l2_val = l2_val )
            
        else:
            x = MobileNet.middle_flow_v2( x, alpha, expansion, dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val )
            
            output_layer = MobileNet.exit_flow_v2( x, alpha = alpha, num_outputs = num_outputs, output_activation = output_activation, 
                                                  pool = pool, base_dropout = base_dropout, top_dropout = top_dropout,
                                                  l1_val = l1_val, l2_val = l2_val )

        return tf.keras.models.Model( input_layer, output_layer )
    
    def get_MobileNetV2(self, input_shape: tuple[int], alpha: float, expansion: int, num_outputs: int, 
                        output_activation: str, pool: bool, base_dropout: float, 
                        top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ MobileNetV2 has: 2 blocks w/  24 filters, 
                             3 blocks w/  32 filters, 
                             4 blocks w/  64 filters,
                             3 blocks w/  96 filters, 
                             3 blocks w/ 160 filters,
                             1 block  w/ 320 filters
        """
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, alpha = alpha, expansion = expansion,
                     num_outputs = num_outputs, output_activation = output_activation, 
                     pool = pool, base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val, v3 = False )
        
        return model
    
    def get_MobileNetV3_Small(self, input_shape: tuple[int], alpha: float, expansion: int, 
                              num_outputs: int, output_activation: str, pool: bool, 
                              base_dropout: float, top_dropout: float, 
                              l1_val: float, l2_val: float) -> Model:
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, alpha = alpha, expansion = expansion,
                     num_outputs = num_outputs, output_activation = output_activation, 
                     pool = pool, base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val, v3 = True, small = True )
        
        return model
    
    def get_MobileNetV3_Large(self, input_shape: tuple[int], alpha: float, expansion: int, 
                              num_outputs: int, output_activation: str, pool: bool, 
                              base_dropout: float, top_dropout: float, 
                              l1_val: float, l2_val: float) -> Model:
        
        # Uses the call function to build the model
        model = self(input_shape = input_shape, alpha = alpha, expansion = expansion,
                     num_outputs = num_outputs, output_activation = output_activation, 
                     pool = pool, base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val, v3 = True, small = False )
        
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
    def apply_activation(x: tf.Tensor, activation: str, block: int, num: int, prefix: str=""):
        if activation.lower() == "relu":
            out = ReLU(name = f"{prefix}block{block}_ReLU_num{num}")(x)
            
        elif activation.lower() == "relu6":
            # Uses ReLU6 instead of ReLU. 
            # ReLU6 clips activation at 6 for positive values
            out = ReLU(6., name = f"{prefix}block{block}_ReLU6_num{num}")(x)
            
        elif activation.lower() == "hard_sigmoid":
            # Uses Hard Sigmoid instead of Sigmoid. 
            # ReLU6 is used to aproximate Sigmoid function.
            l_name = f"{prefix}block{block}_HSigmoid_num{num}"
            x = Rescaling( 1, offset=3., name = f"{l_name}a" )(x)
            x = ReLU(6., name = f"{l_name}b")(x)
            out = Rescaling( 1. / 6., offset=0., name = f"{l_name}c" )(x)
            
        elif activation.lower() == "hard_swish":
            # Implements Hard Swish by multiplying ReLU6 and Hard Sigmoid
            h_sig = MobileNet.apply_activation(x, "hard_sigmoid", block, num, prefix = prefix)
            out = Multiply(name = f"{prefix}block{block}_HSwish_num{num}")([x, h_sig])
        
        else:
            return x
            
        return out
    
    @staticmethod
    def squeeze_excite_block( layer_input: tf.Tensor, filters: int, ratio: float, block: int, num: int ) -> tf.Tensor:
        
        # Pooling
        x = GlobalAveragePooling2D( keepdims = True, name = f"se_block{block}_GAvgPool_num{num}")(layer_input)
        
        # First Conv
        n_filters = MobileNet.make_divisible(filters * ratio, 8)
        x = Conv2D( filters = n_filters, kernel_size = 1, strides = 1, padding = "same", use_bias = True,
                    kernel_initializer = "he_uniform", name = f"se_block{block}_Conv2D_num{num}" )(x)
        x = MobileNet.apply_activation(x, "relu", block = block, num = num, prefix = "se_")
        
        # Second Conv
        x = Conv2D( filters = filters, kernel_size = 1, strides = 1, padding = "same", use_bias = True,
                    kernel_initializer = "he_uniform", name = f"se_block{block}_Conv2D_num{num+1}" )(x)
        x = MobileNet.apply_activation(x, "hard_sigmoid", block = block, num = num, prefix = "se_")
        
        # Multiplication
        out = Multiply(name = f"se_block{block}_Mult_num{num}")([x, layer_input])
        
        return out
    
    @staticmethod
    def conv_bn_relu( x: tf.Tensor, num_filters: int, kernel_size: int, strides: int, padding: str, activation: str, 
                      block: int, num: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        if dropchance > 0:
            x = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num{num}")(x)
        
        x = Conv2D( filters = num_filters, kernel_size = kernel_size, strides = strides, padding = padding,
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num{num}" )(x)
        
        x = BatchNormalization(name = f"block{block}_BN_num{num}")(x)
        
        # Uses ReLU6 instead of ReLU. ReLU6 clips activation at 6 for positive values
        if not activation is None:
            x = MobileNet.apply_activation(x, activation, block = block, num = num)
        
        return x
    
    @staticmethod
    def dw_conv_bn_relu( x: tf.Tensor, kernel_size: int, strides: int, padding: str, activation: str, 
                         block: int, num: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        if dropchance > 0:
            x = SpatialDropout2D(dropchance, name = f"block{block}_dwSDropout_num{num}")(x)
        
        x = DepthwiseConv2D( kernel_size = kernel_size, strides = strides, padding = padding, use_bias = False,
                             depthwise_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                             depthwise_initializer = "he_uniform", name = f"block{block}_dwConv2D_num{num}" )(x)
        
        x = BatchNormalization(name = f"block{block}_dwBN_num{num}")(x)
        
        x = MobileNet.apply_activation(x, activation, block = block, num = num, prefix = "dw_")
        
        return x
    
    @staticmethod
    def bottleneck_res_block( x: tf.Tensor, expansion_filters: int, pointwise_filters: int, kernel_size: int, strides: int, 
                              se_ratio: float, activation: str, block: int, dropchance: float, 
                              l1_val: float, l2_val: float ) -> tf.Tensor:
        
        num = 1
        if block > 2:
            y = MobileNet.conv_bn_relu( x, num_filters = expansion_filters, kernel_size = 1, strides = 1, padding = "same", 
                                        activation = activation, block = block, num = num, dropchance = dropchance, 
                                        l1_val = l1_val, l2_val = l2_val)
            num += 1
            
        y = MobileNet.dw_conv_bn_relu( y if (block > 2) else x, kernel_size = kernel_size, strides = strides, padding = "same", 
                                       activation = activation, block = block, num = 1, dropchance = dropchance, 
                                       l1_val = l1_val, l2_val = l2_val)
        
        if not se_ratio is None:
            y = MobileNet.squeeze_excite_block( layer_input = y, filters = expansion_filters, ratio = se_ratio, 
                                                block = block, num = 1 )
        
        y = MobileNet.conv_bn_relu( y, num_filters = pointwise_filters, kernel_size = 1, strides = 1, padding = "same", 
                                    activation = None, block = block, num = num, dropchance = dropchance, 
                                    l1_val = l1_val, l2_val = l2_val)
        
        in_channels = backend.int_shape(x)[-1]
        if in_channels == pointwise_filters and strides == 1:
            y = Add(name = f"block{block}_Add_num1")([x, y])
        return y
    
    @staticmethod
    def entry_flow( x: tf.Tensor, alpha: float, v3: bool, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        activation = "hard_swish" if v3 else "relu6"
        n_filters  = 16 if v3 else MobileNet.make_divisible(32 * alpha, 8)
        x = MobileNet.conv_bn_relu( x, num_filters = n_filters, kernel_size = 3, strides = 2, padding = "same", 
                                    activation = activation, block = 1, num = 1, dropchance = dropchance, 
                                    l1_val = l1_val, l2_val = l2_val )

        if not v3:
            pointwise_filters = MobileNet.make_divisible(16 * alpha, 8)
            x = MobileNet.bottleneck_res_block( x, expansion_filters = n_filters, pointwise_filters = pointwise_filters, 
                                                kernel_size = 3, strides = 1, se_ratio = None, activation = "relu", 
                                                block = 2, dropchance = dropchance, 
                                                l1_val = l1_val, l2_val = l2_val )
        
        return x
    
    @staticmethod
    def middle_flow_v2( x: tf.Tensor, alpha: float, expansion: int, dropchance: float, 
                        l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # num_filters: num_blocks
        stack_dict = { 24: 2, 32: 3, 64: 4, 96: 3, 160: 3, 320: 1 }
        
        start_block = 3
        for num_filters, num_blocks in stack_dict.items():
            
            final_block = (start_block + num_blocks)
            for block in range( start_block, final_block ):
            
                in_channels = backend.int_shape(x)[-1]
                expansion_filters = expansion * in_channels
                pointwise_filters = MobileNet.make_divisible(alpha * num_filters, 8)

                strides = 2 if (block == start_block) and (not num_filters in [96, 320]) else 1
                x = MobileNet.bottleneck_res_block( x, expansion_filters = expansion_filters, pointwise_filters = pointwise_filters, 
                                                    kernel_size = 3, strides = strides, se_ratio = None, activation = "relu6", 
                                                    block = block, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
                
            start_block += num_blocks
        
        return x
            
    @staticmethod
    def exit_flow_v2( x: tf.Tensor, alpha: float, num_outputs: int, output_activation: str, pool: bool, 
                      base_dropout: float, top_dropout: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        start_block = 19

        # Alpha is only applied to last conv if it's > 1.
        n_filters = 1280
        if alpha > 1.0:
            n_filters = MobileNet.make_divisible(n_filters * alpha, 8)
        
        x = MobileNet.conv_bn_relu( x, num_filters = n_filters, kernel_size = 1, strides = 1, padding = "same", 
                                    activation = "relu6", block = start_block, num = 1, dropchance = base_dropout, 
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
        out = tf.keras.layers.Dense( num_outputs, activation = output_activation, name = "Classification_Layer" )(x)
        
        return out
    
    @staticmethod
    def middle_flow_v3_small( x: tf.Tensor, alpha: float, expansion: int, dropchance: float, 
                              l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Indexes for blocks with 3x3 kernels
        kernel_idx = [i for i in range(3)]
        
        # Indexes for blocks with relu activation
        relu_idx = [i for i in range(3)]
        
        # Indexes for blocks without Squeeze and Excite blocks
        se_idx = [i for i in range(1, 3)]
        
        # Indexes for blocks with stride 2
        stride_idx = [0, 1, 3, 8]
        
        # base_filters: expansion_list
        # Each item for the expansion_list contains the expansion for 1 block
        stack_dict = { 16: [1], 24: [72./16, 88./24], 40: [4, 6, 6], 48: [3, 3], 96: [6, 6, 6] }
                
        x = MobileNet.middle_flow_v3( x, alpha, expansion, stack_dict, stride_idx, se_idx, 
                                     kernel_idx, relu_idx, dropchance, l1_val, l2_val )
        
        return x
    
    @staticmethod
    def middle_flow_v3_large( x: tf.Tensor, alpha: float, expansion: int, dropchance: float, 
                              l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Indexes for blocks with 3x3 kernels
        kernel_idx = [i for i in range(3)] + [i for i in range(6, 12)]
        
        # Indexes for blocks with relu activation
        relu_idx = [i for i in range(6)]
        
        # Indexes for blocks without Squeeze and Excite blocks
        se_idx = [i for i in range(3)] + [i for i in range(6, 10)]
        
        # Indexes for blocks with stride 2
        stride_idx = [1, 3, 6, 12]
        
        # base_filters: expansion
        stack_dict = { 16: [1], 24: [4, 3], 40: [3, 3, 3], 
                       80: [6, 2.5, 2.3, 2.3], 112: [6, 6], 
                       160: [6, 6, 6] }
                
        x = MobileNet.middle_flow_v3( x, alpha, expansion, stack_dict, stride_idx, se_idx, 
                                     kernel_idx, relu_idx, dropchance, l1_val, l2_val )
        
        return x
    
    @staticmethod
    def middle_flow_v3( x: tf.Tensor, alpha: float, expansion: int, stack_dict: dict, stride_idx: list[int], 
                        se_idx: list[int], kernel_idx: list[int], relu_idx: list[int], dropchance: float, 
                        l1_val: float, l2_val: float ) -> tf.Tensor:
        
        block = 2
        for base_filters, exp_list in stack_dict.items():
            for expansion in exp_list:
                
                # Infers kernel_size, activation, se_ration and strides from the given lists
                strides = 2 if  (block-2) in stride_idx else 1
                se_ratio = None if (block-2) in se_idx else 1/4
                kernel_size = 3 if (block-2) in kernel_idx else 5
                activation = "relu" if (block-2) in relu_idx else "hard_swish"
            
                in_channels = backend.int_shape(x)[-1]
                pointwise_filters = MobileNet.make_divisible(alpha * base_filters, 8)
                expansion_filters = MobileNet.make_divisible(expansion * in_channels, 8)
                
                x = MobileNet.bottleneck_res_block( x, expansion_filters = expansion_filters, pointwise_filters = pointwise_filters, 
                                                    kernel_size = kernel_size, strides = strides, se_ratio = se_ratio, 
                                                    activation = activation, block = block, dropchance = dropchance, 
                                                    l1_val = l1_val, l2_val = l2_val )
                block += 1
        
        return x
             
    @staticmethod
    def exit_flow_v3( x: tf.Tensor, alpha: float, expansion: float, small: bool, num_outputs: int, output_activation: str, 
                      pool: bool, base_dropout: float, top_dropout: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        start_block = 13 if small else 17
        
        in_channels = backend.int_shape(x)[-1]
        num_filters = MobileNet.make_divisible(expansion * in_channels, 8)
        x = MobileNet.conv_bn_relu( x, num_filters = num_filters, kernel_size = 1, strides = 1, padding = "same", 
                                    activation = "hard_swish", block = start_block, num = 1, dropchance = base_dropout, 
                                    l1_val = l1_val, l2_val = l2_val )
        
        # Global pooling used
        if pool.lower() == "max":
            x = GlobalMaxPooling2D(keepdims = True, name = "GlobalMaxPooling")(x)
        else:
            x = GlobalAveragePooling2D(keepdims = True, name = "GlobalAvgPooling")(x)

        # Alpha is only applied to last conv if it's > 1.
        num_filters = 1024 if small else 1280
        if alpha > 1.0:
            num_filters = MobileNet.make_divisible(num_filters * alpha, 8)
        
        x = Conv2D( filters = num_filters, kernel_size = 1, strides = 1, padding = "same", use_bias = True,
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                    kernel_initializer = "he_uniform", name = f"block{start_block}_Conv2D_num{2}" )(x)
        
        x = MobileNet.apply_activation(x, "hard_swish", block = start_block, num = 2)

        # Adds Dropout before the output dense layer if top_dropout > 0
        if top_dropout > 0:
            x = tf.keras.layers.Dropout(top_dropout, name = f"topDropout_num1")(x)
        
        x = Conv2D( filters = num_outputs, kernel_size = 1, strides = 1, padding = "same", use_bias = True,
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                    kernel_initializer = "he_uniform", name = f"Logits" )(x)
        x = Flatten( name = f"Flatten" )(x)
        
        # Adds output layer
        out = Activation( activation = output_activation, name = "Classification_Layer" )(x)
        
        return out