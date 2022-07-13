import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D

class ResNet:

    def __init__(self):
        return
    
    def __call__( self, num_block_list: list[int], input_shape: tuple[int], 
                  num_outputs: int, output_activation: str, pool: bool, 
                  base_dropout: float, top_dropout: float, l1_val: float, 
                  l2_val: float, use_bottleneck: bool ) -> Model:

        # Model's input layer
        input_layer = tf.keras.layers.Input( shape = input_shape, name = "Input" )

        # First block
        num_filters = 64
        x = Conv2D( filters = num_filters, kernel_size = 7, strides = 2, padding = "same", 
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                    kernel_initializer = "he_uniform", name = "block1_Conv2D_num1" )(input_layer)
        x = MaxPooling2D(3, strides = 2, name = "block1_MaxPooling2D_num1")(x)
        x = ResNet.bn_relu(x, block = 1, num = 1)
        
        # Remaining blocks up until the last one
        current_block = 2
        for idx, num_blocks in enumerate(num_block_list):

            # Sets the strides in the last block to 1
            last_block_strides = 2 if (idx+1 < len(num_block_list)) else 1
            
            # Inserts a stack of 'num_blocks' residual blocks
            if use_bottleneck:
                x = ResNet.residual_bottleneck_stack_v2(stack_input = x, n_filters = num_filters, 
                                                n_blocks = num_blocks, start_block = current_block,
                                                fstrides = last_block_strides, dropchance = base_dropout, 
                                                l1_val = l1_val, l2_val = l2_val)
            else:
                x = ResNet.residual_stack_v2(stack_input = x, n_filters = num_filters, 
                                             n_blocks = num_blocks, start_block = current_block,
                                             fstrides = last_block_strides, dropchance = base_dropout, 
                                             l1_val = l1_val, l2_val = l2_val)
                
            
            num_filters *= 2
            current_block += num_blocks
            
        # BN + ReLU for the final block
        x = ResNet.bn_relu(x, block = current_block, num = 1)
        
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
    
    def get_ResNet18( self, input_shape: tuple[int], num_outputs: int, 
                      output_activation: str, pool: bool, base_dropout: float, 
                      top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ ResNet18 has: 2 blocks w/  64 filters, 2 blocks w/ 128 filters, 
                          2 blocks w/ 256 filters, 2 blocks w/ 512 filters
        """
        # List with the number of blocks for each stack
        num_blocks = [ 2, 2, 2, 2 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, input_shape = input_shape, 
                  num_outputs = num_outputs, output_activation = output_activation, 
                  pool = pool, base_dropout = base_dropout, top_dropout = top_dropout, 
                  l1_val = l1_val, l2_val = l2_val, use_bottleneck = False )
        
        return model
    
    def get_ResNet34( self, input_shape: tuple[int], num_outputs: int, 
                      output_activation: str, pool: bool, base_dropout: float, 
                      top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ ResNet34 has: 3 blocks w/  64 filters, 4 blocks w/ 128 filters, 
                          6 blocks w/ 256 filters, 3 blocks w/ 512 filters
        """
        # List with the number of blocks for each stack
        num_blocks = [ 3, 4, 6, 3 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, input_shape = input_shape, 
                  num_outputs = num_outputs, output_activation = output_activation, 
                  pool = pool, base_dropout = base_dropout, top_dropout = top_dropout, 
                  l1_val = l1_val, l2_val = l2_val, use_bottleneck = False )
        
        return model
    
    def get_ResNet50( self, input_shape: tuple[int], num_outputs: int, 
                      output_activation: str, pool: bool, base_dropout: float, 
                      top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ ResNet50 has: 3 blocks w/  64 filters, 4 blocks w/ 128 filters, 
                          6 blocks w/ 256 filters, 3 blocks w/ 512 filters
                This architecture also replaces the two 3x3 convolutional layers in the
            residual path for a bottleneck module, which is a sequence of 3 convolutional
            layers, respectively 1x1, 3x3, 1x1.
        """
        # List with the number of blocks for each stack
        num_blocks = [ 3, 4, 6, 3 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, input_shape = input_shape, 
                  num_outputs = num_outputs, output_activation = output_activation, 
                  pool = pool, base_dropout = base_dropout, top_dropout = top_dropout, 
                  l1_val = l1_val, l2_val = l2_val, use_bottleneck = True )
        
        return model
    
    def get_ResNet101( self, input_shape: tuple[int], num_outputs: int, 
                      output_activation: str, pool: bool, base_dropout: float, 
                      top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ ResNet101 has:  3 blocks w/  64 filters, 4 blocks w/ 128 filters, 
                           23 blocks w/ 256 filters, 3 blocks w/ 512 filters
                This architecture also replaces the two 3x3 convolutional layers in the
            residual path for a bottleneck module, which is a sequence of 3 convolutional
            layers, respectively 1x1, 3x3, 1x1.
        """
        # List with the number of blocks for each stack
        num_blocks = [ 3, 4, 23, 3 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, input_shape = input_shape, 
                  num_outputs = num_outputs, output_activation = output_activation, 
                  pool = pool, base_dropout = base_dropout, top_dropout = top_dropout, 
                  l1_val = l1_val, l2_val = l2_val, use_bottleneck = True )
        
        return model
    
    def get_ResNet152( self, input_shape: tuple[int], num_outputs: int, 
                      output_activation: str, pool: bool, base_dropout: float, 
                      top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ ResNet152 has:  3 blocks w/  64 filters, 8 blocks w/ 128 filters, 
                           36 blocks w/ 256 filters, 3 blocks w/ 512 filters
                This architecture also replaces the two 3x3 convolutional layers in the
            residual path for a bottleneck module, which is a sequence of 3 convolutional
            layers, respectively 1x1, 3x3, 1x1.
        """
        # List with the number of blocks for each stack
        num_blocks = [ 3, 8, 36, 3 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, input_shape = input_shape, 
                  num_outputs = num_outputs, output_activation = output_activation, 
                  pool = pool, base_dropout = base_dropout, top_dropout = top_dropout, 
                  l1_val = l1_val, l2_val = l2_val, use_bottleneck = True )
        
        return model
    
    @staticmethod
    def bn_relu( input: tf.Tensor, block: int, num: int, bn_first: bool = True ) -> tf.Tensor:
        x = BatchNormalization(name = f"block{block}_BN_num{num}")(input)
        x = ReLU(name = f"block{block}_ReLU_num{num}")(x)
        return x
            
    @staticmethod
    def residual_block_v2( x: tf.Tensor, n_filters: int, kernel_size: int, layer_strides: int, conv_shortcut: bool,
                           block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Residual path of the ResNet block. Bias are removed as BatchNorm layer would remove them anyway...
        if block > 2:
            y = ResNet.bn_relu(x, block = block, num = 1)
        y = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num1")(y if block > 2 else x)
        y = Conv2D( filters = n_filters, kernel_size = kernel_size, strides = layer_strides, padding = "same", 
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num1" )(y)
        
        y = ResNet.bn_relu(y, block = block, num = 2 if block > 2 else 1)
        y = Conv2D( filters = n_filters, kernel_size = kernel_size, strides = 1, padding = "same", 
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num2" )(y)

        # Skip-connection path (shortcut) of the ResNet block
        if conv_shortcut:
            x = Conv2D( filters = n_filters, kernel_size = 1, strides = 1, padding = "same", 
                        kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                        kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num0" )(x)

        elif layer_strides > 1:
            x = MaxPooling2D(1, strides = 2, name = f"block{block}_MaxPooling2D_num1")(x)

        out = Add(name = f"block{block}_Add_num1")([x, y])
        return out
    
    @staticmethod
    def residual_stack_v2(stack_input: tf.Tensor, n_filters: int, n_blocks: int, fstrides: int,
                          start_block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Number of the last block
        final_block = start_block + n_blocks - 1
        
        # First block in the stack, has a 1x1 convolutional shortcut to increase the number of feature maps
        x = ResNet.residual_block_v2( x = stack_input, n_filters = n_filters, kernel_size = 3, 
                                      layer_strides = 1, conv_shortcut = True, block = start_block, 
                                      dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        
        # Sequence of intermediate blocks, have identity shortcut
        for i in range(start_block+1, final_block):
            x = ResNet.residual_block_v2( x = x, n_filters = n_filters, kernel_size = 3, 
                                          layer_strides = 1, conv_shortcut = False, block = i, 
                                          dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        
        # Final block, has a MaxPooling2D shortcut to reduce the feature maps dims
        x = ResNet.residual_block_v2( x = x, n_filters = n_filters, kernel_size = 3, 
                                      layer_strides = fstrides, conv_shortcut = False, block = final_block, 
                                      dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        return x
            
    @staticmethod
    def residual_bottleneck_block_v2( x: tf.Tensor, n_filters: int, kernel_size: int, layer_strides: int, conv_shortcut: bool,
                                      block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Residual path of the ResNet block. Bias are removed as BatchNorm layer would remove them anyway...
        if block > 2:
            y = ResNet.bn_relu(x, block = block, num = 1)
        y = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num1")(y if block > 2 else x)
        y = Conv2D( filters = n_filters, kernel_size = 1, strides = 1, padding = "same",
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num1" )(y)
        
        y = ResNet.bn_relu(y, block = block, num = 2 if block > 2 else 1)
        y = Conv2D( filters = n_filters, kernel_size = kernel_size, strides = layer_strides, padding = "same", 
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num2" )(y)
        
        y = ResNet.bn_relu(y, block = block, num = 3 if block > 2 else 2)
        y = Conv2D( filters = 4 * n_filters, kernel_size = 1, strides = 1, padding = "same", 
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num3" )(y)

        # Skip-connection path (shortcut) of the ResNet block
        if conv_shortcut:
            x = Conv2D( filters = 4 * n_filters, kernel_size = 1, strides = 1, padding = "same", 
                        kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val),
                        kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num0" )(x)

        elif layer_strides > 1:
            x = MaxPooling2D(1, strides = 2, name = f"block{block}_MaxPooling2D_num1")(x)

        out = Add(name = f"block{block}_Add_num1")([x, y])
        return out
    
    @staticmethod
    def residual_bottleneck_stack_v2(stack_input: tf.Tensor, n_filters: int, n_blocks: int, fstrides: int,
                                     start_block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Number of the last block
        final_block = start_block + n_blocks - 1
        
        # First block in the stack, has a 1x1 convolutional shortcut to increase the number of feature maps
        x = ResNet.residual_bottleneck_block_v2( x = stack_input, n_filters = n_filters, kernel_size = 3, 
                                                       layer_strides = 1, conv_shortcut = True, block = start_block, 
                                                       dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        
        # Sequence of intermediate blocks, have identity shortcut
        for i in range(start_block+1, final_block):
            x = ResNet.residual_bottleneck_block_v2( x = x, n_filters = n_filters, kernel_size = 3, 
                                                           layer_strides = 1, conv_shortcut = False, block = i, 
                                                           dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        
        # Final block, has a MaxPooling2D shortcut to reduce the feature maps dims
        x = ResNet.residual_bottleneck_block_v2( x = x, n_filters = n_filters, kernel_size = 3, 
                                                       layer_strides = fstrides, conv_shortcut = False, block = final_block, 
                                                       dropchance = dropchance, l1_val = l1_val, l2_val = l2_val )
        return x