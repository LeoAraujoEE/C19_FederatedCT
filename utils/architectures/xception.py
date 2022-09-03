import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D

class Xception:

    def __init__(self):
        return
    
    def __call__( self, num_middle_blocks: int, input_shape: tuple[int], num_outputs: int, 
                  output_activation: str, pool: bool, base_dropout: float, top_dropout: float, 
                  l1_val: float, l2_val: float ) -> Model:

        # Model's input layer
        input_layer = tf.keras.layers.Input( shape = input_shape, name = "Input" )
        
        # Entry flow
        x = Xception.entry_flow( x = input_layer, dropchance = 0, 
                                l1_val = l1_val, l2_val = l2_val )
        
        # Middle flow: Adds blocks according to 'num_middle_blocks'
        final_block = 5 + num_middle_blocks
        for n_block in range(5, final_block):
            x = Xception.block_b(x, block = n_block, dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val)
        
        # Exit flow
        x = Xception.exit_flow( x, block = final_block, dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val )
        
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
    
    def get_Xception( self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool,
                      base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ Xception repeats the middle block 8 times to build the Middle flow
        """
        
        # Uses the call function to build the model
        model = self(num_middle_blocks = 8, input_shape = input_shape, num_outputs = num_outputs, 
                     output_activation = output_activation, pool = pool, base_dropout = base_dropout, 
                     top_dropout = top_dropout, l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    @staticmethod
    def conv_bn_relu( x: tf.Tensor, num_filters: int, kernel_size: tuple[int], strides: int, padding: str, 
                      block: int, num: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        if dropchance > 0:
            x = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num{num}")(x)
        
        x = Conv2D( filters = num_filters, kernel_size = kernel_size, strides = strides, padding = padding,
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num{num}" )(x)
        
        x = BatchNormalization(name = f"block{block}_BN_num{num}")(x)
        x = ReLU(name = f"block{block}_ReLU_num{num}")(x)
        
        return x
    
    @staticmethod
    def sepconv_bn( x: tf.Tensor, num_filters: int, kernel_size: tuple[int], block: int, num: int, 
                    dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        if dropchance > 0:
            x = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num{num}")(x)
        
        x = SeparableConv2D( filters = num_filters, kernel_size = kernel_size, strides = 1, padding = "same",
                    depthwise_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), 
                    pointwise_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), 
                    depthwise_initializer = "he_uniform", pointwise_initializer = "he_uniform", 
                    use_bias = False, name = f"block{block}_Conv2D_num{num}" )(x)
        
        x = BatchNormalization(name = f"block{block}_BN_num{num}")(x)
        
        return x
    
    @staticmethod
    def block_a( x: tf.Tensor, num_filters: int, block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        """ Residual Block that consists on 2 SeparableConvs with BN and ReLU followed by a MaxPooling layer.
        This block is used in the entry flow and also at the exit flow. The residual path uses a 1x1 Conv shortcut.
        """
        
        # Residual path
        y = Xception.sepconv_bn( x, num_filters, kernel_size = (3,3), block = block, num = 1, 
                                 dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        
        y = ReLU(name = f"block{block}_ReLU_num1")(y)
        y = Xception.sepconv_bn( y, num_filters, kernel_size = (3,3), block = block, num = 2, 
                                 dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        
        y = MaxPooling2D((3,3), strides = 2, padding = "same", name = f"block{block}_MaxPooling2D_num1")(y)

        # Skip-connection path (shortcut)
        x = Conv2D( filters = num_filters, kernel_size = 1, strides = 2, padding = "same", use_bias = False, 
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num0" )(x)
        x = BatchNormalization(name = f"block{block}_BN_num0")(x)
        
        # Output
        out = Add(name = f"block{block}_Add_num1")([x, y])
        return out
    
    @staticmethod
    def block_b( x: tf.Tensor, block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        """ Residual Block that consists on 3 SeparableConvs with BN and ReLU.
        This block is used in the middle flow and also at the exit flow.
        """
        
        # Residual path
        y = ReLU(name = f"block{block}_ReLU_num1")(x)
        y = Xception.sepconv_bn( y, 728, kernel_size = (3,3), block = block, num = 1, dropchance = dropchance, 
                                 l1_val = l1_val, l2_val = l2_val )
        
        y = ReLU(name = f"block{block}_ReLU_num2")(y)
        y = Xception.sepconv_bn( y, 728, kernel_size = (3,3), block = block, num = 2, dropchance = dropchance, 
                                 l1_val = l1_val, l2_val = l2_val )
        
        y = ReLU(name = f"block{block}_ReLU_num3")(y)
        y = Xception.sepconv_bn( y, 728, kernel_size = (3,3), block = block, num = 3, dropchance = dropchance, 
                                 l1_val = l1_val, l2_val = l2_val )
        
        # Output
        out = Add(name = f"block{block}_Add_num1")([x, y])
        return out
    
    @staticmethod
    def entry_flow( x: tf.Tensor, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Block 1
        x = Xception.conv_bn_relu( x, num_filters = 32, kernel_size = (3,3), strides = 2, padding = "valid", 
                                   block = 1, num = 1, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        x = Xception.conv_bn_relu( x, num_filters = 64, kernel_size = (3,3), strides = 1, padding = "valid", 
                                   block = 1, num = 2, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        for b, n_filters in enumerate([128, 256, 728]):
            x = Xception.block_a( x, num_filters = n_filters, block = (b+2), dropchance = dropchance, 
                                  l1_val = l1_val, l2_val = l2_val )
        
        return x
    
    @staticmethod
    def exit_flow( x: tf.Tensor, block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        """ Residual Block that consists on 2 SeparableConvs with BN and ReLU followed by a MaxPooling layer.
        This block is used in the entry flow and also at the exit flow. The residual path uses a 1x1 Conv shortcut.
        """
        
        # Residual path
        y = Xception.sepconv_bn( x, 728, kernel_size = (3,3), block = block, num = 1, 
                                 dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        
        y = ReLU(name = f"block{block}_ReLU_num1")(y)
        y = Xception.sepconv_bn( y, 1024, kernel_size = (3,3), block = block, num = 2, 
                                 dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        
        y = MaxPooling2D((3,3), strides = 2, padding = "same", name = f"block{block}_MaxPooling2D_num1")(y)

        # Skip-connection path (shortcut)
        x = Conv2D( filters = 1024, kernel_size = 1, strides = 2, padding = "same", use_bias = False, 
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num0" )(x)
        x = BatchNormalization(name = f"block{block}_BN_num0")(x)
        
        # Addition of residual + shortcut
        out = Add(name = f"block{block}_Add_num1")([x, y])
        
        # Final separable convs+bn+relu
        out = Xception.sepconv_bn(out, 1536, kernel_size = (3,3), block = block, num = 3, 
                                  dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        out = ReLU(name = f"block{block}_ReLU_num2")(out)
        
        out = Xception.sepconv_bn(out, 2048, kernel_size = (3,3), block = block, num = 4, 
                                  dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        out = ReLU(name = f"block{block}_ReLU_num3")(out)
        return out