import numpy as np
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

class Inception:

    def __init__(self):
        return
    
    def __call__( self, num_block_list: list[int], input_shape: tuple[int], num_outputs: int, 
                  output_activation: str, pool: bool, bn_scaling: bool, v3: bool, 
                  base_dropout: float, top_dropout: float, l1_val: float, l2_val: float ) -> Model:

        # Model's input layer
        input_layer = tf.keras.layers.Input( shape = input_shape, name = "Input" )
        
        # Block 1
        if v3:
            x = Inception.inception_v3_stem( x = input_layer, bn_scaling = bn_scaling, dropchance = base_dropout, 
                                             l1_val = l1_val, l2_val = l2_val )
        else:
            x = Inception.inception_v4_stem( x = input_layer, bn_scaling = bn_scaling, dropchance = base_dropout, 
                                             l1_val = l1_val, l2_val = l2_val )
            
        
        # Stage indicates what kind of inception block is used
        for stage in range(3):
            
            # Gets the numbers for the starting and final blocks for the current stage
            start_block = 2 if (stage == 0) else (final_block+1)
            final_block = start_block + num_block_list[stage]
            
            # Adds blocks according to 'num_block_list'
            for n_block in range(start_block, final_block):
                
                # Uses type a blocks for the first stage
                if stage == 0:
                    x = Inception.inception_block_a(x, bn_scaling = bn_scaling, v3 = v3, block = n_block, 
                                                    dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val)
                    continue
                
                # Uses type b blocks for the second stage
                if stage == 1:
                    # For InceptionV3, the number of intermediate filters changes according to the block in the 2nd stage, 
                    # being 128 for the first block, 160 for those in the middle, and 192 for the last block
                    filters = 128 if n_block == start_block else (160 if n_block < (final_block - 1) else 192)
                    
                    x = Inception.inception_block_b(x, middle_filters = filters, bn_scaling = bn_scaling, v3 = v3, 
                                                    block = n_block, dropchance = base_dropout, 
                                                    l1_val = l1_val, l2_val = l2_val)
                    continue
                
                # Uses type c blocks for the third stage
                if stage == 2:
                    x = Inception.inception_block_c(x, bn_scaling = bn_scaling, v3 = v3, block = n_block, 
                                                    dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val)
                    continue
            
            # A reduction block is applied between stages
            if stage == 0:
                # Uses type a blocks for the first stage
                x = Inception.inception_reduction_block_a(x, bn_scaling = bn_scaling, v3 = v3, block = final_block, 
                                                          l1_val = l1_val, l2_val = l2_val)
                continue
            
            if stage == 1:
                # Uses type b blocks for the second stage
                x = Inception.inception_reduction_block_b(x, bn_scaling = bn_scaling, v3 = v3, block = final_block, 
                                                          l1_val = l1_val, l2_val = l2_val)
                continue
        
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
    
    def get_InceptionV3( self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool,
                       base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ InceptionV3 has 3 stages with: 3 blocks in the 1st stage
                                           4 blocks in the 2nd stage
                                           2 blocks in the 3rd stage
                                         
            A special downsampling block is used between subsequent stages
        """
        # List with the number of blocks for each stack
        num_blocks = [ 3, 4, 2 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, input_shape = input_shape, num_outputs = num_outputs, 
                     output_activation = output_activation, pool = pool, bn_scaling = True, v3 = True, 
                     base_dropout = base_dropout, top_dropout = top_dropout, l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    def get_InceptionV4( self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool,
                       base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ Inception has 3 stages with: 4 blocks in the 1st stage
                                         7 blocks in the 2nd stage
                                         3 blocks in the 3rd stage
                                         
            A special downsampling block is used between subsequent stages
        """
        # List with the number of blocks for each stack
        num_blocks = [ 4, 7, 3 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, input_shape = input_shape, num_outputs = num_outputs, 
                     output_activation = output_activation, pool = pool, bn_scaling = True, v3 = False, 
                     base_dropout = base_dropout, top_dropout = top_dropout, l1_val = l1_val, l2_val = l2_val )
        
        return model
    
    @staticmethod
    def conv_bn_relu( l_input: tf.Tensor, num_filters: int, kernel_size: tuple[int], strides: int, padding: str, 
                      bn_scaling: bool, block: int, num: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        if dropchance > 0:
            x = SpatialDropout2D(dropchance, name = f"block{block}_SDropout_num{num}")(l_input)
        
        x = Conv2D( filters = num_filters, kernel_size = kernel_size, strides = strides, padding = padding,
                    kernel_regularizer = tf.keras.regularizers.L1L2(l1 = l1_val, l2 = l2_val), use_bias = False,
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num{num}" )(x if dropchance > 0 else l_input)
        
        x = BatchNormalization(scale = bn_scaling, name = f"block{block}_BN_num{num}")(x)
        x = ReLU(name = f"block{block}_ReLU_num{num}")(x)
        
        return x
    
    @staticmethod
    def inception_v3_stem( x: tf.Tensor, bn_scaling: bool, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Block 1
        x = Inception.conv_bn_relu( x, num_filters = 32, kernel_size = (3,3), strides = 2, padding = "valid", 
                                    bn_scaling = bn_scaling, block = 1, num = 1, dropchance = dropchance, 
                                    l1_val = l1_val, l2_val = l2_val)
        
        x = Inception.conv_bn_relu( x, num_filters = 32, kernel_size = (3,3), strides = 1, padding = "valid", bn_scaling = bn_scaling, 
                                    block = 1, num = 2, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        x = Inception.conv_bn_relu( x, num_filters = 64, kernel_size = (3,3), strides = 1, padding = "same", bn_scaling = bn_scaling, 
                                    block = 1, num = 3, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        x = MaxPooling2D( (3, 3), strides = (2, 2), padding = "valid", name = f"block1_MaxPooling2D_num1")(x)
        
        x = Inception.conv_bn_relu( x, num_filters = 80, kernel_size = (1,1), strides = 1, padding = "valid", bn_scaling = bn_scaling, 
                                    block = 1, num = 4, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        x = Inception.conv_bn_relu( x, num_filters = 192, kernel_size = (3,3), strides = 1, padding = "valid", bn_scaling = bn_scaling, 
                                   block = 1, num = 5, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        out = MaxPooling2D( (3, 3), strides = (2, 2), padding = "valid", name = f"block1_MaxPooling2D_num2")(x)
        
        return out
    
    @staticmethod
    def inception_v4_stem( x: tf.Tensor, bn_scaling: bool, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # Block 1
        x = Inception.conv_bn_relu( x, num_filters = 32, kernel_size = (3,3), strides = 2, padding = "valid", 
                                    bn_scaling = bn_scaling, block = 1, num = 1, dropchance = dropchance, 
                                    l1_val = l1_val, l2_val = l2_val)
        
        x = Inception.conv_bn_relu( x, num_filters = 32, kernel_size = (3,3), strides = 1, padding = "valid", bn_scaling = bn_scaling, 
                                    block = 1, num = 2, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        x = Inception.conv_bn_relu( x, num_filters = 64, kernel_size = (3,3), strides = 1, padding = "same", bn_scaling = bn_scaling, 
                                    block = 1, num = 3, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        # MaxPooling branch
        b_pool = MaxPooling2D( (3, 3), strides = (2, 2), padding = "valid", name = f"block1_MaxPooling2D_num1")(x)
        
        # 3x3 Conv branch
        b_conv = Inception.conv_bn_relu( x, num_filters = 96, kernel_size = (3,3), strides = 2, padding = "valid", bn_scaling = bn_scaling, 
                                         block = 1, num = 4, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        x = Concatenate(name = f"block1_Concatenate_num1")([b_pool, b_conv])
        
        # 7x7x3 Conv Branch
        b_7x7x3 = Inception.conv_bn_relu( x, num_filters = 64, kernel_size = (1,1), strides = 1, padding = "same", 
                                          bn_scaling = bn_scaling, block = 1, num = 5, dropchance = dropchance, 
                                          l1_val = l1_val, l2_val = l2_val)
        
        b_7x7x3 = Inception.conv_bn_relu( b_7x7x3, num_filters = 64, kernel_size = (1,7), strides = 1, padding = "same", 
                                          bn_scaling = bn_scaling, block = 1, num = 6, dropchance = dropchance, 
                                          l1_val = l1_val, l2_val = l2_val)
        
        b_7x7x3 = Inception.conv_bn_relu( b_7x7x3, num_filters = 64, kernel_size = (7,1), strides = 1, padding = "same", 
                                          bn_scaling = bn_scaling, block = 1, num = 7, dropchance = dropchance, 
                                          l1_val = l1_val, l2_val = l2_val)
        
        b_7x7x3 = Inception.conv_bn_relu( b_7x7x3, num_filters = 96, kernel_size = (3,3), strides = 1, padding = "valid", 
                                          bn_scaling = bn_scaling, block = 1, num = 8, dropchance = dropchance, 
                                          l1_val = l1_val, l2_val = l2_val)
        
        # 3x3 Conv Branch
        b_3x3 = Inception.conv_bn_relu( x, num_filters = 64, kernel_size = (1,1), strides = 1, padding = "same", bn_scaling = bn_scaling, 
                                        block = 1, num = 9, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        b_3x3 = Inception.conv_bn_relu( x, num_filters = 96, kernel_size = (3,3), strides = 1, padding = "valid", bn_scaling = bn_scaling, 
                                        block = 1, num = 10, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        x = Concatenate(name = f"block1_Concatenate_num2")([b_7x7x3, b_3x3])
        
        # MaxPooling branch
        b_pool = MaxPooling2D( (3, 3), strides = (2, 2), padding = "valid", name = f"block1_MaxPooling2D_num2")(x)
        
        # 3x3 Conv branch
        b_conv = Inception.conv_bn_relu( x, num_filters = 192, kernel_size = (3,3), strides = 2, padding = "valid", bn_scaling = bn_scaling, 
                                         block = 1, num = 11, dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        out = Concatenate(name = f"block1_Concatenate_num3")([b_pool, b_conv])
        
        return out
    
    @staticmethod
    def inception_block_a( x: tf.Tensor, bn_scaling: bool, v3: bool, block: int, 
                           dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # 1x1 Conv2D Branch
        f1x1 = 64 if v3 else 96
        branch_1x1 = Inception.conv_bn_relu( x, num_filters = f1x1, kernel_size = (1,1), strides = 1, padding = "same", 
                                            bn_scaling = bn_scaling, block = block, num = 1, dropchance = dropchance, 
                                            l1_val = l1_val, l2_val = l2_val)
        if v3:
            # 5x5 Conv2D Branch
            branch_5x5 = Inception.conv_bn_relu( x, num_filters = 48, kernel_size = (1,1), strides = 1, padding = "same", 
                                                bn_scaling = bn_scaling, block = block, num = 2, dropchance = dropchance, 
                                                l1_val = l1_val, l2_val = l2_val)
            
            branch_5x5 = Inception.conv_bn_relu( branch_5x5, num_filters = 64, kernel_size = (5,5), strides = 1, padding = "same", 
                                                bn_scaling = bn_scaling, block = block, num = 3, dropchance = dropchance, 
                                                l1_val = l1_val, l2_val = l2_val)
            
        else:
            # Inception V4 replaces 5x5 Conv2D for 2 3x3 Convs
            branch_5x5 = Inception.conv_bn_relu( x, num_filters = 64, kernel_size = (1,1), strides = 1, padding = "same", 
                                                bn_scaling = bn_scaling, block = block, num = 2, dropchance = dropchance, 
                                                l1_val = l1_val, l2_val = l2_val)
            
            branch_5x5 = Inception.conv_bn_relu( branch_5x5, num_filters = 96, kernel_size = (3, 3), strides = 1, padding = "same", 
                                                bn_scaling = bn_scaling, block = block, num = 3, dropchance = dropchance, 
                                                l1_val = l1_val, l2_val = l2_val)
        
        # Double 3x3 Branch
        branch_3x3_dbl = Inception.conv_bn_relu( x, num_filters = 64, kernel_size = (1,1), strides = 1, padding = "same", 
                                                bn_scaling = bn_scaling, block = block, num = 4, dropchance = dropchance, 
                                                l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = Inception.conv_bn_relu( branch_3x3_dbl, num_filters = 96, kernel_size = (3,3), strides = 1, padding = "same", 
                                                bn_scaling = bn_scaling, block = block, num = 5, dropchance = dropchance, 
                                                l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = Inception.conv_bn_relu( branch_3x3_dbl, num_filters = 96, kernel_size = (3,3), strides = 1, padding = "same", 
                                                bn_scaling = bn_scaling, block = block, num = 6, dropchance = dropchance, 
                                                l1_val = l1_val, l2_val = l2_val)
        
        # Average Pooling Branch
        branch_avg_pool = AveragePooling2D( (3, 3), strides = (1, 1), padding = "same", name = f"block{block}_AvgPooling2D_num1")(x)
        
        n_filters_v3 = 32 if (block == 2) else 64
        n_filters = n_filters_v3 if v3 else 96
        branch_avg_pool = Inception.conv_bn_relu( x, num_filters = n_filters, kernel_size = (1,1), strides = 1, padding = "same", 
                                                  bn_scaling = bn_scaling, block = block, num = 7, dropchance = dropchance, 
                                                  l1_val = l1_val, l2_val = l2_val)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num1")([branch_1x1, branch_5x5, branch_3x3_dbl, branch_avg_pool])
        
        return out
    
    @staticmethod
    def inception_block_b( x: tf.Tensor, middle_filters: int, bn_scaling: bool, v3: bool, block: int, 
                           dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # 1x1 Conv2D Branch
        f1x1 = 192 if v3 else 384
        branch_1x1 = Inception.conv_bn_relu( x, num_filters = f1x1, kernel_size = (1,1), strides = 1, padding = "same", 
                                            bn_scaling = bn_scaling, block = block, num = 1, dropchance = dropchance, 
                                            l1_val = l1_val, l2_val = l2_val)
        
        # 7x7 Conv2D Branch
        f7x7 = middle_filters if v3 else 192
        branch_7x7 = Inception.conv_bn_relu( x, num_filters = f7x7, kernel_size = (1,1), strides = 1, padding = "same", 
                                             bn_scaling = bn_scaling, block = block, num = 2, dropchance = dropchance, 
                                             l1_val = l1_val, l2_val = l2_val)
        
        f7x7 = middle_filters if v3 else 224
        branch_7x7 = Inception.conv_bn_relu( branch_7x7, num_filters = f7x7, kernel_size = (1,7), strides = 1, padding = "same", 
                                             bn_scaling = bn_scaling, block = block, num = 3, dropchance = dropchance, 
                                             l1_val = l1_val, l2_val = l2_val)
        
        f7x7 = 192 if v3 else 256
        branch_7x7 = Inception.conv_bn_relu( branch_7x7, num_filters = f7x7, kernel_size = (7,1), strides = 1, padding = "same", 
                                             bn_scaling = bn_scaling, block = block, num = 4, dropchance = dropchance, 
                                             l1_val = l1_val, l2_val = l2_val)
        
        # Double 7x7 Conv2D Branch
        f7x7_dbl = middle_filters if v3 else 192
        branch_7x7_dbl = Inception.conv_bn_relu( x, num_filters = f7x7_dbl, kernel_size = (1,1), strides = 1, padding = "same", 
                                                 bn_scaling = bn_scaling, block = block, num = 5, dropchance = dropchance, 
                                                 l1_val = l1_val, l2_val = l2_val)
        
        branch_7x7_dbl = Inception.conv_bn_relu( branch_7x7_dbl, num_filters = f7x7_dbl, kernel_size = (7,1), strides = 1, 
                                                 padding = "same", bn_scaling = bn_scaling, block = block, num = 6, 
                                                 dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        f7x7_dbl = middle_filters if v3 else 224
        branch_7x7_dbl = Inception.conv_bn_relu( branch_7x7_dbl, num_filters = f7x7_dbl, kernel_size = (1,7), strides = 1, 
                                                 padding = "same", bn_scaling = bn_scaling, block = block, num = 7, 
                                                 dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        branch_7x7_dbl = Inception.conv_bn_relu( branch_7x7_dbl, num_filters = f7x7_dbl, kernel_size = (7,1), strides = 1, 
                                                 padding = "same", bn_scaling = bn_scaling, block = block, num = 8, 
                                                 dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        f7x7_dbl = 192 if v3 else 256
        branch_7x7_dbl = Inception.conv_bn_relu( branch_7x7_dbl, num_filters = f7x7_dbl, kernel_size = (1,7), strides = 1, 
                                                 padding = "same", bn_scaling = bn_scaling, block = block, num = 9, 
                                                 dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        # Average Pooling Branch
        fpool = 192 if v3 else 128
        branch_avg_pool = AveragePooling2D( (3, 3), strides = (1, 1), padding = "same", name = f"block{block}_AvgPooling2D_num1")(x)
        branch_avg_pool = Inception.conv_bn_relu( x, num_filters = fpool, kernel_size = (1,1), strides = 1, padding = "same", 
                                                 bn_scaling = bn_scaling, block = block, num = 10, dropchance = dropchance, 
                                                 l1_val = l1_val, l2_val = l2_val)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num1")([branch_1x1, branch_7x7, branch_7x7_dbl, branch_avg_pool])
        
        return out
    
    @staticmethod
    def inception_block_c( x: tf.Tensor, bn_scaling: bool, v3: bool, block: int, 
                           dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # 1x1 Conv2D Branch
        f1x1 = 320 if v3 else 256
        branch_1x1 = Inception.conv_bn_relu( x, num_filters = f1x1, kernel_size = (1,1), strides = 1, padding = "same", 
                                            bn_scaling = bn_scaling, block = block, num = 1, dropchance = dropchance, 
                                            l1_val = l1_val, l2_val = l2_val)
        
        # 3x3 Conv2D Branch
        branch_3x3 = Inception.conv_bn_relu( x, num_filters = 384, kernel_size = (1,1), strides = 1, padding = "same", 
                                             bn_scaling = bn_scaling, block = block, num = 2, dropchance = dropchance, 
                                             l1_val = l1_val, l2_val = l2_val)
        
        f3x3 = 384 if v3 else 256
        branch_3x3_a = Inception.conv_bn_relu( branch_3x3, num_filters = f3x3, kernel_size = (1,3), strides = 1, padding = "same", 
                                             bn_scaling = bn_scaling, block = block, num = 3, dropchance = dropchance, 
                                             l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_b = Inception.conv_bn_relu( branch_3x3, num_filters = f3x3, kernel_size = (3,1), strides = 1, padding = "same", 
                                             bn_scaling = bn_scaling, block = block, num = 4, dropchance = dropchance, 
                                             l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3 = Concatenate(name = f"block{block}_Concatenate_num1")([branch_3x3_a, branch_3x3_b])
        
        # Double 3x3 Conv2D Branch
        f3x3_dbl = 448 if v3 else 384
        branch_3x3_dbl = Inception.conv_bn_relu( x, num_filters = f3x3_dbl, kernel_size = (1,1), strides = 1, padding = "same", 
                                                 bn_scaling = bn_scaling, block = block, num = 5, dropchance = dropchance, 
                                                 l1_val = l1_val, l2_val = l2_val)
        if v3:
            branch_3x3_dbl = Inception.conv_bn_relu( branch_3x3_dbl, num_filters = 384, kernel_size = (3,3), strides = 1, 
                                                    padding = "same", bn_scaling = bn_scaling, block = block, num = 6, 
                                                    dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
            num = 7
            
        else:
            branch_3x3_dbl = Inception.conv_bn_relu( branch_3x3_dbl, num_filters = 448, kernel_size = (1,3), strides = 1, 
                                                    padding = "same", bn_scaling = bn_scaling, block = block, num = 6, 
                                                    dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
            
            branch_3x3_dbl = Inception.conv_bn_relu( branch_3x3_dbl, num_filters = 512, kernel_size = (3,1), strides = 1, 
                                                    padding = "same", bn_scaling = bn_scaling, block = block, num = 7, 
                                                    dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
            num = 8
        
        f3x3_dbl = 384 if v3 else 256
        branch_3x3_dbl_a = Inception.conv_bn_relu( branch_3x3_dbl, num_filters = f3x3_dbl, kernel_size = (1,3), strides = 1, 
                                                 padding = "same", bn_scaling = bn_scaling, block = block, num = num, 
                                                 dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        num += 1
        
        branch_3x3_dbl_b = Inception.conv_bn_relu( branch_3x3_dbl, num_filters = f3x3_dbl, kernel_size = (3,1), strides = 1, 
                                                 padding = "same", bn_scaling = bn_scaling, block = block, num = num, 
                                                 dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        num += 1
        
        branch_3x3_dbl = Concatenate(name = f"block{block}_Concatenate_num2")([branch_3x3_dbl_a, branch_3x3_dbl_b])
        
        # Average Pooling Branch
        fpool = 192 if v3 else 256
        branch_avg_pool = AveragePooling2D( (3, 3), strides = (1, 1), padding = "same", name = f"block{block}_AvgPooling2D_num1")(x)
        branch_avg_pool = Inception.conv_bn_relu( x, num_filters = fpool, kernel_size = (1,1), strides = 1, padding = "same", 
                                                 bn_scaling = bn_scaling, block = block, num = num, dropchance = dropchance, 
                                                 l1_val = l1_val, l2_val = l2_val)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num3")([branch_1x1, branch_3x3, branch_3x3_dbl, branch_avg_pool])
        
        return out
    
    @staticmethod
    def inception_reduction_block_a( x: tf.Tensor, bn_scaling: bool, v3: bool, block: int, 
                                     l1_val: float, l2_val: float ) -> tf.Tensor:
        
        if v3:
            # Values for InceptionV3
            k, l, m, n = 64, 96, 96, 384
            
        else:
            # Values for InceptionV4
            k, l, m, n = 192, 224, 256, 384
        
        # 3x3 Conv2D Branch
        branch_3x3 = Inception.conv_bn_relu( x, num_filters = n, kernel_size = (3,3), strides = 2, padding = "valid", 
                                            bn_scaling = bn_scaling, block = block, num = 1, dropchance = 0, 
                                            l1_val = l1_val, l2_val = l2_val)
        
        # Double 3x3 Branch
        branch_3x3_dbl = Inception.conv_bn_relu( x, num_filters = k, kernel_size = (1,1), strides = 1, padding = "same", 
                                                 bn_scaling = bn_scaling, block = block, num = 2, dropchance = 0, 
                                                 l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = Inception.conv_bn_relu( branch_3x3_dbl, num_filters = l, kernel_size = (3,3), strides = 1, padding = "same", 
                                                 bn_scaling = bn_scaling, block = block, num = 3, dropchance = 0, 
                                                 l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = Inception.conv_bn_relu( branch_3x3_dbl, num_filters = m, kernel_size = (3,3), strides = 2, padding = "valid", 
                                                 bn_scaling = bn_scaling, block = block, num = 4, dropchance = 0, 
                                                 l1_val = l1_val, l2_val = l2_val)
        
        # Max Pooling Branch
        branch_avg_pool = MaxPooling2D( (3, 3), strides = (2, 2), padding = "valid", name = f"block{block}_MaxPooling2D_num1")(x)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num1")([branch_3x3, branch_3x3_dbl, branch_avg_pool])
        
        return out
        
    @staticmethod
    def inception_reduction_block_b( x: tf.Tensor, bn_scaling: bool, v3: bool, block: int, 
                                     l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # 3x3 Conv2D Branch
        branch_3x3 = Inception.conv_bn_relu( x, num_filters = 192, kernel_size = (1,1), strides = 1, padding = "same", 
                                            bn_scaling = bn_scaling, block = block, num = 1, dropchance = 0,
                                            l1_val = l1_val, l2_val = l2_val)
        
        f3x3 = 320 if v3 else 192
        branch_3x3 = Inception.conv_bn_relu( branch_3x3, num_filters = f3x3, kernel_size = (3,3), strides = 2, padding = "valid", 
                                             bn_scaling = bn_scaling, block = block, num = 2, dropchance = 0, 
                                             l1_val = l1_val, l2_val = l2_val)
        
        # 7x7x3 Branch
        f7x7x3 = 192 if v3 else 256
        branch_7x7x3 = Inception.conv_bn_relu( x, num_filters = f7x7x3, kernel_size = (1,1), strides = 1, padding = "same", 
                                               bn_scaling = bn_scaling, block = block, num = 3, dropchance = 0, 
                                               l1_val = l1_val, l2_val = l2_val)
        
        branch_7x7x3 = Inception.conv_bn_relu( branch_7x7x3, num_filters = f7x7x3, kernel_size = (1,7), strides = 1, padding = "same", 
                                               bn_scaling = bn_scaling, block = block, num = 4, dropchance = 0, 
                                               l1_val = l1_val, l2_val = l2_val)
        
        f7x7x3 = 192 if v3 else 320
        branch_7x7x3 = Inception.conv_bn_relu( branch_7x7x3, num_filters = f7x7x3, kernel_size = (7,1), strides = 1, padding = "same", 
                                               bn_scaling = bn_scaling, block = block, num = 5, dropchance = 0, 
                                               l1_val = l1_val, l2_val = l2_val)
        
        branch_7x7x3 = Inception.conv_bn_relu( branch_7x7x3, num_filters = f7x7x3, kernel_size = (3,3), strides = 2, padding = "valid", 
                                               bn_scaling = bn_scaling, block = block, num = 6, dropchance = 0, 
                                               l1_val = l1_val, l2_val = l2_val)
        
        # Max Pooling Branch
        branch_avg_pool = MaxPooling2D( (3, 3), strides = (2, 2), padding = "valid", name = f"block{block}_MaxPooling2D_num1")(x)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num1")([branch_3x3, branch_7x7x3, branch_avg_pool])
        
        return out