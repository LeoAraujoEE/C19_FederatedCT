import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D

from utils.architectures.inception import Inception

class InceptionResNet(Inception):

    def __init__(self):
        return
    
    def __call__( self, num_block_list: list[int], input_shape: tuple[int], num_outputs: int, 
                  output_activation: str, pool: bool, bn_scaling: bool, base_dropout: float, 
                  top_dropout: float, l1_val: float, l2_val: float ) -> Model:

        # Model's input layer
        input_layer = tf.keras.layers.Input( shape = input_shape, 
                                             name = "Input" )
        
        # Block 1
        x = self.inception_v3_stem( x = input_layer, bn_scaling = bn_scaling,
                                    dropchance = 0, l1_val = l1_val, 
                                    l2_val = l2_val )
        
        # Block 2 - According to Keras, InceptionResNet_v2 starts with a Inception Block A from Inception V3
        x = InceptionResNet.inception_v3_block_a( x, bn_scaling = bn_scaling,
                                        block = 2, dropchance = base_dropout,
                                        l1_val = l1_val, l2_val = l2_val )
            
        # Stage indicates what kind of inception block is used
        for stage, block_type in enumerate(["a", "b", "c"]):
            
            # Gets the numbers for the starting and final blocks for the current stage
            start_block = 3 if (stage == 0) else (final_block+1)
            final_block = start_block + num_block_list[stage]
            
            # Reduces final block by 1 for last stage as this block is placed outside the loop
            if stage == 2:
                final_block -= 1
            
            # Adds blocks according to 'num_block_list'
            for n_block in range(start_block, final_block):
                
                # Blocks of the first stage have 0.17 scaling for residual connections
                if stage == 0:
                    scaling = .17
                
                # Blocks of the second stage have 0.10 scaling for residual connections
                elif stage == 1:
                    scaling = .1
                    
                # Blocks of the third stage have 0.2 scaling for residual connections
                else:
                    scaling = .2
                
                # Uses type a blocks for the first stage
                x = InceptionResNet.block(layer_input = x, bn_scaling = bn_scaling, scaling = scaling, block = n_block, 
                                          block_type = block_type, dropchance = base_dropout, l1_val = l1_val, l2_val = l2_val)
            
            # A different reduction block is applied in the end of each stage
            x = InceptionResNet.reduction_block(x, bn_scaling = bn_scaling, block = final_block, stage = stage, 
                                                l1_val = l1_val, l2_val = l2_val)
        
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
    
    def get_InceptionResNetV2( self, input_shape: tuple[int], num_outputs: int, output_activation: str, pool: bool,
                               base_dropout: float, top_dropout: float, l1_val: float, l2_val: float) -> Model:
        
        """ InceptionResNetV2 has 3 stages with: 10 blocks in the 1st stage
                                                 20 blocks in the 2nd stage
                                                 10 blocks in the 3rd stage
                                         
            A special downsampling block is used between subsequent stages
        """
        # List with the number of blocks for each stack
        num_blocks = [ 10, 20, 10 ]
        
        # Uses the call function to build the model
        model = self(num_block_list = num_blocks, input_shape = input_shape, num_outputs = num_outputs, 
                     output_activation = output_activation, pool = pool, bn_scaling = False, 
                     base_dropout = base_dropout, top_dropout = top_dropout, 
                     l1_val = l1_val, l2_val = l2_val )
        
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
    def inception_v3_block_a( x: tf.Tensor, bn_scaling: bool, block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        """ Inception Block-A based on InceptionV3, which is used by Keras InceptionResNetV2 model """
        
        # 1x1 Conv2D Branch
        branch_1x1 = InceptionResNet.conv_bn_relu( x, num_filters = 96, kernel_size = (1,1), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 1, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
            
        # 5x5 Conv2D Branch
        branch_5x5 = InceptionResNet.conv_bn_relu( x, num_filters = 48, kernel_size = (1,1), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 2, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        branch_5x5 = InceptionResNet.conv_bn_relu( branch_5x5, num_filters = 64, kernel_size = (5,5), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 3, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        # Double 3x3 Branch
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( x, num_filters = 64, kernel_size = (1,1), strides = 1, padding = "same", 
                                                       bn_scaling = bn_scaling, block = block, num = 4, dropchance = dropchance, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( branch_3x3_dbl, num_filters = 96, kernel_size = (3,3), strides = 1, padding = "same", 
                                                       bn_scaling = bn_scaling, block = block, num = 5, dropchance = dropchance, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( branch_3x3_dbl, num_filters = 96, kernel_size = (3,3), strides = 1, padding = "same", 
                                                       bn_scaling = bn_scaling, block = block, num = 6, dropchance = dropchance, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        # Average Pooling Branch
        branch_avg_pool = AveragePooling2D( (3, 3), strides = (1, 1), padding = "same", name = f"block{block}_AvgPooling2D_num1")(x)
        
        branch_avg_pool = InceptionResNet.conv_bn_relu( x, num_filters = 64, kernel_size = (1,1), strides = 1, padding = "same", 
                                                        bn_scaling = bn_scaling, block = block, num = 7, dropchance = dropchance, 
                                                        l1_val = l1_val, l2_val = l2_val)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num1")([branch_1x1, branch_5x5, branch_3x3_dbl, branch_avg_pool])
        
        return out
    
    @staticmethod
    def block( layer_input: tf.Tensor, bn_scaling: bool, scaling: float, block: int, block_type: str, 
               dropchance: float, l1_val: float, l2_val: float, activation: bool = True ) -> tf.Tensor:
        """ Inception Resnet Block """
        
        if block_type.lower() == "a":
            num_conv_bn = 7
            num_filters = 320
            x = InceptionResNet.block_a( x = layer_input, bn_scaling = bn_scaling, block = block, 
                                         dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
            
        elif block_type.lower() == "b":
            num_conv_bn = 5
            num_filters = 1088
            x = InceptionResNet.block_b( x = layer_input, bn_scaling = bn_scaling, block = block, 
                                         dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
            
        elif block_type.lower() == "c":
            num_conv_bn = 5
            num_filters = 2080
            x = InceptionResNet.block_c( x = layer_input, bn_scaling = bn_scaling, block = block, 
                                         dropchance = dropchance, l1_val = l1_val, l2_val = l2_val)
        
        # Conv2D with bias and no ReLU/BN
        x = Conv2D( filters = num_filters, kernel_size = (1, 1), strides = 1, padding = "same",
                    kernel_initializer = "he_uniform", name = f"block{block}_Conv2D_num{num_conv_bn}" )(x)
        
        x = Rescaling( scaling, name = f"block{block}_Scaling_num1" )(x)
        
        out = Add(name = f"block{block}_Add_num1")([layer_input, x])

        if activation:
            out = ReLU(name = f"block{block}_ReLU_num{num_conv_bn}")(out)
        
        return out
    
    @staticmethod
    def block_a( x: tf.Tensor, bn_scaling: bool, block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # 1x1 Conv2D Branch
        branch_1x1 = InceptionResNet.conv_bn_relu( x, num_filters = 32, kernel_size = (1,1), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 1, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        # InceptionResNet V4 replaces 5x5 Conv2D for 2 3x3 Convs
        branch_3x3 = InceptionResNet.conv_bn_relu( x, num_filters = 32, kernel_size = (1,1), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 2, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3 = InceptionResNet.conv_bn_relu( branch_3x3, num_filters = 32, kernel_size = (3, 3), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 3, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        # Double 3x3 Branch
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( x, num_filters = 32, kernel_size = (1,1), strides = 1, padding = "same", 
                                                       bn_scaling = bn_scaling, block = block, num = 4, dropchance = dropchance, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( branch_3x3_dbl, num_filters = 48, kernel_size = (3,3), strides = 1, padding = "same", 
                                                       bn_scaling = bn_scaling, block = block, num = 5, dropchance = dropchance, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( branch_3x3_dbl, num_filters = 64, kernel_size = (3,3), strides = 1, padding = "same", 
                                                       bn_scaling = bn_scaling, block = block, num = 6, dropchance = dropchance, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num1")([branch_1x1, branch_3x3, branch_3x3_dbl])
        
        return out
    
    @staticmethod
    def block_b( x: tf.Tensor, bn_scaling: bool, block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # 1x1 Conv2D Branch
        branch_1x1 = InceptionResNet.conv_bn_relu( x, num_filters = 192, kernel_size = (1,1), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 1, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        # 7x7 Conv2D Branch
        branch_7x7 = InceptionResNet.conv_bn_relu( x, num_filters = 128, kernel_size = (1,1), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 2, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        branch_7x7 = InceptionResNet.conv_bn_relu( branch_7x7, num_filters = 160, kernel_size = (1,7), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 3, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        branch_7x7 = InceptionResNet.conv_bn_relu( branch_7x7, num_filters = 192, kernel_size = (7,1), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 4, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num1")([branch_1x1, branch_7x7])
        
        return out
    
    @staticmethod
    def block_c( x: tf.Tensor, bn_scaling: bool, block: int, dropchance: float, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # 1x1 Conv2D Branch
        branch_1x1 = InceptionResNet.conv_bn_relu( x, num_filters = 192, kernel_size = (1,1), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 1, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        # 3x3 Conv2D Branch
        branch_3x3 = InceptionResNet.conv_bn_relu( x, num_filters = 192, kernel_size = (1,1), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 2, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3 = InceptionResNet.conv_bn_relu( branch_3x3, num_filters = 224, kernel_size = (1,3), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 3, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3 = InceptionResNet.conv_bn_relu( branch_3x3, num_filters = 256, kernel_size = (3,1), strides = 1, padding = "same", 
                                                   bn_scaling = bn_scaling, block = block, num = 4, dropchance = dropchance, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num1")([branch_1x1, branch_3x3])
        
        return out
    
    @staticmethod
    def reduction_block( x: tf.Tensor, bn_scaling: bool, block: int, stage: int, l1_val: float, l2_val: float ) -> tf.Tensor:
        """ Inception Resnet Reduction Block """
        
        if stage == 0:
            # Uses type a blocks for the first stage
            out = InceptionResNet.reduction_block_a(x, bn_scaling = bn_scaling, block = block, 
                                                    l1_val = l1_val, l2_val = l2_val)
            
        elif stage == 1:
            # Uses type b blocks for the second stage
            out = InceptionResNet.reduction_block_b(x, bn_scaling = bn_scaling, block = block, 
                                                    l1_val = l1_val, l2_val = l2_val)
            
        elif stage == 2:
            # Uses type b blocks for the second stage
            out = InceptionResNet.reduction_block_c(x, bn_scaling = bn_scaling, block = block, 
                                                    l1_val = l1_val, l2_val = l2_val)
            
        return out
    
    @staticmethod
    def reduction_block_a( x: tf.Tensor, bn_scaling: bool, block: int, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # 3x3 Conv2D Branch
        branch_3x3 = InceptionResNet.conv_bn_relu( x, num_filters = 384, kernel_size = (3,3), strides = 2, padding = "valid", 
                                                   bn_scaling = bn_scaling, block = block, num = 1, dropchance = 0, 
                                                   l1_val = l1_val, l2_val = l2_val)
        
        # Double 3x3 Branch
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( x, num_filters = 256, kernel_size = (1,1), strides = 1, padding = "same", 
                                                       bn_scaling = bn_scaling, block = block, num = 2, dropchance = 0, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( branch_3x3_dbl, num_filters = 256, kernel_size = (3,3), strides = 1, padding = "same", 
                                                       bn_scaling = bn_scaling, block = block, num = 3, dropchance = 0, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( branch_3x3_dbl, num_filters = 384, kernel_size = (3,3), strides = 2, padding = "valid", 
                                                       bn_scaling = bn_scaling, block = block, num = 4, dropchance = 0, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        # Max Pooling Branch
        branch_avg_pool = MaxPooling2D( (3, 3), strides = (2, 2), padding = "valid", name = f"block{block}_MaxPooling2D_num1")(x)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num1")([branch_3x3, branch_3x3_dbl, branch_avg_pool])
        
        return out
        
    @staticmethod
    def reduction_block_b( x: tf.Tensor, bn_scaling: bool, block: int, l1_val: float, l2_val: float ) -> tf.Tensor:
        
        # 3x3 Conv2D Branch A
        branch_3x3a = InceptionResNet.conv_bn_relu( x, num_filters = 256, kernel_size = (1,1), strides = 1, padding = "same", 
                                                    bn_scaling = bn_scaling, block = block, num = 1, dropchance = 0,
                                                    l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3a = InceptionResNet.conv_bn_relu( branch_3x3a, num_filters = 384, kernel_size = (3,3), strides = 2, padding = "valid", 
                                                    bn_scaling = bn_scaling, block = block, num = 2, dropchance = 0, 
                                                    l1_val = l1_val, l2_val = l2_val)
        
        # 3x3 Conv2D Branch B
        branch_3x3b = InceptionResNet.conv_bn_relu( x, num_filters = 256, kernel_size = (1,1), strides = 1, padding = "same", 
                                                    bn_scaling = bn_scaling, block = block, num = 3, dropchance = 0,
                                                    l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3b = InceptionResNet.conv_bn_relu( branch_3x3b, num_filters = 288, kernel_size = (3,3), strides = 2, padding = "valid", 
                                                    bn_scaling = bn_scaling, block = block, num = 4, dropchance = 0, 
                                                    l1_val = l1_val, l2_val = l2_val)
        
        # 3x3 Conv2D Branch Double
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( x, num_filters = 256, kernel_size = (1,1), strides = 1, padding = "same", 
                                                       bn_scaling = bn_scaling, block = block, num = 5, dropchance = 0, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( branch_3x3_dbl, num_filters = 288, kernel_size = (3,3), strides = 1, padding = "same", 
                                                       bn_scaling = bn_scaling, block = block, num = 6, dropchance = 0, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        branch_3x3_dbl = InceptionResNet.conv_bn_relu( branch_3x3_dbl, num_filters = 320, kernel_size = (3,3), strides = 2, padding = "valid", 
                                                       bn_scaling = bn_scaling, block = block, num = 7, dropchance = 0, 
                                                       l1_val = l1_val, l2_val = l2_val)
        
        # Max Pooling Branch
        branch_max_pool = MaxPooling2D( (3, 3), strides = (2, 2), padding = "valid", name = f"block{block}_MaxPooling2D_num1")(x)
        
        # Concatenation
        out = Concatenate(name = f"block{block}_Concatenate_num1")([branch_3x3a, branch_3x3b, branch_3x3_dbl, branch_max_pool])
        
        return out
        
    @staticmethod
    def reduction_block_c( x: tf.Tensor, bn_scaling: bool, block: int, l1_val: float, l2_val: float ) -> tf.Tensor:
                
        # Uses type a blocks for the first stage
        x = InceptionResNet.block(layer_input = x, bn_scaling = bn_scaling, scaling = 1., block = block, 
                                  block_type = "c", dropchance = 0, l1_val = l1_val, l2_val = l2_val, 
                                  activation = False)
        
        # Final Conv2D x BN x ReLU
        x = InceptionResNet.conv_bn_relu( x, num_filters = 1536, kernel_size = (1,1), strides = 1, padding = "same", 
                                          bn_scaling = bn_scaling, block = block, num = 6, dropchance = 0, 
                                          l1_val = l1_val, l2_val = l2_val)
        
        return x