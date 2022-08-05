import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D

class MobileNet:

    def __init__(self):
        return
    
    def __call__( self, stack_dict: int, input_shape: tuple[int], alpha: float, expansion: int,
                  num_outputs: int, output_activation: str, pool: bool, base_dropout: float, top_dropout: float, 
                  l1_val: float, l2_val: float ) -> Model:
        
        # Example for MobileNetV2
        # stack_dict = { 24: 2, 32: 3, 64: 4, 96: 3, 160: 3 }

        # Model's input layer
        input_layer = tf.keras.layers.Input( shape = input_shape, name = "Input" )
        
        ###########################################################################################
        # Put the model here ----------------------------------------------------------------------
        
        # Entry Flow:
        x = MobileNet.entry_flow_v2(input_layer, alpha = alpha, dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val )
        
        # Remaining blocks up until the last one
        current_block = 3
        for num_filters, num_blocks in stack_dict.items():
            x = MobileNet.bottleneck_res_stack(x, base_filters = num_filters, alpha = alpha, expansion = expansion,
                                               start_block = current_block, final_block = (current_block + num_blocks),
                                               dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val)
            current_block += num_blocks
        
        # Entry Flow:
        x = MobileNet.exit_flow_v2(x, alpha = alpha, expansion = expansion, start_block = current_block, 
                                   dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val )
        
        # Put the model here ----------------------------------------------------------------------
        ###########################################################################################
        
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
    
    def get_MobileNetV2(self, input_shape: tuple[int], alpha: float, expansion: int, num_outputs: int, 
                        output_activation: str, pool: bool, base_dropout: float, 
                        top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ MobileNetV2 has: 2 blocks w/  24 filters, 
                             3 blocks w/  32 filters, 
                             4 blocks w/  64 filters,
                             3 blocks w/  96 filters, 
                             3 blocks w/ 160 filters
        """
        # List with the number channels and their respective number of blocks for each stack
        stack_dict = { 24: 2, 32: 3, 64: 4, 96: 3, 160: 3 }
        
        # Uses the call function to build the model
        model = self(stack_dict = stack_dict, input_shape = input_shape, alpha = alpha, expansion = expansion,
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
    def conv_bn_relu( x: tf.Tensor, num_filters: int, kernel_size: int, strides: int, padding: str, activation: bool, 
                      block: int, num: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        if dropchance > 0:
            x = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num{num}")(x)
        
        x = Conv2D( filters = num_filters, kernel_size = kernel_size, strides = strides, padding = padding,
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num{num}" )(x)
        
        x = BatchNormalization(name = f"block{block}_BN_num{num}")(x)
        
        # Uses ReLU6 instead of ReLU. ReLU6 clips activation at 6 for positive values
        if activation:
            x = ReLU(6., name = f"block{block}_ReLU_num{num}")(x)
        
        return x
    
    @staticmethod
    def dw_conv_bn_relu( x: tf.Tensor, strides: int, padding: str, block: int, num: int, 
                         dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        if dropchance > 0:
            x = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num{num}")(x)
        
        x = DepthwiseConv2D( kernel_size = 3, strides = strides, padding = padding, use_bias = False,
                             depthwise_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                             depthwise_initializer = "he_uniform", name = f"block{block}_DWConv2D_num{num}" )(x)
        
        x = BatchNormalization(name = f"block{block}_BN_num{num}")(x)
        
        # Uses ReLU6 instead of ReLU. ReLU6 clips activation at 6 for positive values
        x = ReLU(6., name = f"block{block}_ReLU_num{num}")(x)
        
        return x
    
    @staticmethod
    def bottleneck_res_block( x: tf.Tensor, expansion_filters: int, pointwise_filters: int, strides: int, block: int, 
                              dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        num = 1
        if block > 2:
            y = MobileNet.conv_bn_relu( x, num_filters = expansion_filters, kernel_size = 1, strides = 1, padding = "same", 
                                        activation = True, block = block, num = num, dropchance = dropchance, 
                                        l1_val = l1_val, l2_val = l2_val)
            num += 1
            
        # TODO: Check ZeroPadding
        y = MobileNet.dw_conv_bn_relu( y if (block > 2) else x, strides = strides, padding = "same", block = block, num = num, 
                                       dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        num += 1
        
        y = MobileNet.conv_bn_relu( y, num_filters = pointwise_filters, kernel_size = 1, strides = 1, padding = "same", 
                                    activation = False, block = block, num = num, dropchance = dropchance, 
                                    l1_val = l1_val, l2_val = l2_val)
        
        in_channels = backend.int_shape(x)[-1]
        if in_channels == pointwise_filters and strides == 1:
            y = Add(name = f"block{block}_Add_num1")([x, y])
        return y
    
    @staticmethod
    def bottleneck_res_stack( x: tf.Tensor, base_filters: int, alpha: float, expansion: int, start_block: int, final_block: int, 
                              dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        for num in range( start_block, final_block ):
        
            in_channels = backend.int_shape(x)[-1]
            expansion_filters = expansion * in_channels
            pointwise_filters = MobileNet.make_divisible(alpha * base_filters, 8)

            strides = 2 if (num == start_block) and (base_filters != 96) else 1
            x = MobileNet.bottleneck_res_block( x, expansion_filters = expansion_filters, pointwise_filters = pointwise_filters, 
                                                strides = strides, block = num, dropchance = dropchance, 
                                                l1_val = l1_val, l2_val = l2_val )
        
        return x
    
    @staticmethod
    def entry_flow_v2( x: tf.Tensor, alpha: float, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        n_filters = MobileNet.make_divisible(32 * alpha, 8)
        x = MobileNet.conv_bn_relu( x, num_filters = n_filters, kernel_size = 3, strides = 2, padding = "same", 
                                    activation = True, block = 1, num = 1, dropchance = dropchance, 
                                    l1_val = l1_val, l2_val = l2_val )

        pointwise_filters = MobileNet.make_divisible(16 * alpha, 8)
        x = MobileNet.bottleneck_res_block( x, expansion_filters = n_filters, pointwise_filters = pointwise_filters, 
                                            strides = 1, block = 2, dropchance = dropchance, 
                                            l1_val = l1_val, l2_val = l2_val )
        
        return x
    
    @staticmethod
    def exit_flow_v2( x: tf.Tensor, alpha: float, expansion: int, start_block: int,
                      dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        in_channels = backend.int_shape(x)[-1]
        expansion_filters = expansion * in_channels
        pointwise_filters = MobileNet.make_divisible(alpha * 320, 8)
        
        x = MobileNet.bottleneck_res_block( x, expansion_filters = expansion_filters, pointwise_filters = pointwise_filters, 
                                            strides = 1, block = start_block, dropchance = dropchance, 
                                            l1_val = l1_val, l2_val = l2_val )

        # Alpha is only applied to last conv if it's > 1.
        n_filters = 1280
        if alpha > 1.0:
            n_filters = MobileNet.make_divisible(n_filters * alpha, 8)
        
        block = start_block + 1
        x = MobileNet.conv_bn_relu( x, num_filters = n_filters, kernel_size = 1, strides = 1, padding = "same", 
                                    activation = True, block = block, num = 1, dropchance = dropchance, 
                                    l1_val = l1_val, l2_val = l2_val )
        
        return x