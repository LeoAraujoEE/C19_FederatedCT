import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D

class DenseNet:

    def __init__(self):
        return
    
    def __call__( self, num_block_list: list, growth_rate: int, reduction_factor: float,
                  input_shape: tuple, num_outputs: int, output_activation: str, pool: bool, 
                  base_dropout: float, top_dropout: float, l1_val: float, l2_val: float ) -> Model:

        # Model's input layer
        input_layer = tf.keras.layers.Input( shape = input_shape, name = "Input" )

        # First block
        num_filters = 64
        x = Conv2D( filters = num_filters, kernel_size = 7, strides = 2, padding = "same", 
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                    kernel_initializer = "he_uniform", name = "block1_Conv2D_num1", use_bias = False )(input_layer)
        x = DenseNet.bn_relu(x, block = 1, num = 1)
        x = MaxPooling2D(3, strides = 2, padding = "same", name = "block1_MaxPooling2D_num1")(x)
        
        # Remaining blocks up until the last one
        current_block = 2
        for idx, num_blocks in enumerate(num_block_list):
            x = DenseNet.dense_stack(x = x, n_new_filters = growth_rate, n_blocks = num_blocks, start_block = current_block, 
                                     dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val)
            current_block += num_blocks
            num_filters += growth_rate * num_blocks
            
            if (idx + 1) < len(num_block_list):
                num_filters = int(num_filters * reduction_factor)
                x = DenseNet.transition_block( x = x, num_filters = num_filters, block = current_block, 
                                               l1_val = l1_val, l2_val = l2_val)
                current_block += 1
            
        # BN + ReLU for the final block
        x = DenseNet.bn_relu(x, block = current_block, num = 1)
        
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
    
    def get_DenseNet121( self, input_shape: tuple, num_outputs: int, 
                         output_activation: str, pool: bool, base_dropout: float, 
                         top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ DenseNet121 has:  6 convolutional blocks in the 1st Dense Stack
                             12 convolutional blocks in the 2nd Dense Stack
                             24 convolutional blocks in the 3rd Dense Stack
                             16 convolutional blocks in the 4th Dense Stack
        """
        # List with the number of blocks for each stack
        num_blocks = [ 6, 12, 24, 16 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, growth_rate = 32, reduction_factor = 0.5,
                     input_shape = input_shape, num_outputs = num_outputs, 
                     output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_DenseNet169( self, input_shape: tuple, num_outputs: int, 
                         output_activation: str, pool: bool, base_dropout: float, 
                         top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ DenseNet121 has:  6 convolutional blocks in the 1st Dense Stack
                             12 convolutional blocks in the 2nd Dense Stack
                             32 convolutional blocks in the 3rd Dense Stack
                             32 convolutional blocks in the 4th Dense Stack
        """
        # List with the number of blocks for each stack
        num_blocks = [ 6, 12, 32, 32 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, growth_rate = 32, reduction_factor = 0.5,
                     input_shape = input_shape, num_outputs = num_outputs, 
                     output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_DenseNet201( self, input_shape: tuple, num_outputs: int, 
                         output_activation: str, pool: bool, base_dropout: float, 
                         top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ DenseNet121 has:  6 convolutional blocks in the 1st Dense Stack
                             12 convolutional blocks in the 2nd Dense Stack
                             48 convolutional blocks in the 3rd Dense Stack
                             32 convolutional blocks in the 4th Dense Stack
        """
        # List with the number of blocks for each stack
        num_blocks = [ 6, 12, 48, 32 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, growth_rate = 32, reduction_factor = 0.5,
                     input_shape = input_shape, num_outputs = num_outputs, 
                     output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_DenseNet264( self, input_shape: tuple, num_outputs: int, 
                         output_activation: str, pool: bool, base_dropout: float, 
                         top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ DenseNet121 has:  6 convolutional blocks in the 1st Dense Stack
                             12 convolutional blocks in the 2nd Dense Stack
                             64 convolutional blocks in the 3rd Dense Stack
                             48 convolutional blocks in the 4th Dense Stack
        """
        # List with the number of blocks for each stack
        num_blocks = [ 6, 12, 64, 48 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, growth_rate = 32, reduction_factor = 0.5,
                     input_shape = input_shape, num_outputs = num_outputs, 
                     output_activation = output_activation, pool = pool, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    @staticmethod
    def bn_relu( input: tf.Tensor, block: int, num: int ) -> tf.Tensor:
        x = BatchNormalization(name = f"block{block}_BN_num{num}")(input)
        x = ReLU(name = f"block{block}_ReLU_num{num}")(x)
        return x
    
    @staticmethod
    def conv_block( x: tf.Tensor, n_new_filters: int, kernel_size: int, block: int, 
                    dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        
        # Convolutional block inside Dense Stacks
        y = DenseNet.bn_relu(x, block = block, num = 1)
        if dropchance > 0:
            y = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num1")(y)
        y = Conv2D( filters = 4 * n_new_filters, kernel_size = 1, strides = 1, padding = "same",
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num1" )(y)
        
        y = DenseNet.bn_relu(y, block = block, num = 2)
        if dropchance > 0:
            y = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num2")(y)
        y = Conv2D( filters = n_new_filters, kernel_size = kernel_size, strides = 1, padding = "same", 
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num2" )(y)

        out = Concatenate(name = f"block{block}_Concatenate_num1")([x, y])
        return out
    
    @staticmethod
    def transition_block( x: tf.Tensor, num_filters: int, block: int, 
                          l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Trasition blocks to downsample data between Dense Stacks
        y = DenseNet.bn_relu(x, block = block, num = 1)
            
        y = Conv2D( filters = num_filters, kernel_size = 1, strides = 1, padding = "same",
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num1" )(y)
        
        y = AveragePooling2D(2, strides = 2, name = f"block{block}_AvgPooling2D_num1")(y)
        
        return y
    
    @staticmethod
    def dense_stack(x: tf.Tensor, n_new_filters: int, n_blocks: int, start_block: int, 
                    dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Number of the last block
        final_block = start_block + n_blocks
        
        # Sequence of intermediate blocks, have identity shortcut
        for i in range(start_block, final_block):
            x = DenseNet.conv_block( x = x, n_new_filters = n_new_filters, kernel_size = 3, 
                                    block = i, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        return x