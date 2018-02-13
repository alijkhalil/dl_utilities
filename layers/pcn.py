# Import statements and global variables
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings, math
import tensorflow as tf

import keras.backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.initializers import Constant


INIT_MULTIPLER=3.5


''' 
These are custom layers for use in an PCN cell.

Their names are geared towards use in an PCN cell 
and therefore, may not describe their functionality 
in the most general way.
'''

# Ensures that the sum of the squares of activations in each channel is 1        
def NormalizePerChannel(name=None):
    def func(x):
        num_channels = K.int_shape(x)[-1]
        ch_tensors = []
        
        for i in range(num_channels):
            div_val = K.sqrt(K.sum(K.square(x[:, :, :, i]), axis=(1, 2), keepdims=True))            
            new_channels = K.expand_dims(tf.divide(x[:, :, :, i], div_val))
            
            ch_tensors.append(new_channels)
            
        return K.concatenate(ch_tensors)
    
    return Lambda(func, name=name)

    
# Custom Tanh layer (with 'scalar' variable being learnable) 
class GetTanhValue(Layer):
    def __init__(self, init_multupler=INIT_MULTIPLER, **kwargs):
        self.init_multupler = init_multupler
        
        super(GetTanhValue, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create trainable scalar and bias variables for the layer.
        self.scalar = self.add_weight(name='multipler', 
                                      shape=(1,),
                                      initializer=Constant(value=self.init_multupler),
                                      trainable=True)
         
        self.bias = self.add_weight(name='bias_term', 
                                      shape=(1,),
                                      initializer='zeros',
                                      trainable=True)
                                      
        super(GetTanhValue, self).build(input_shape)

    def call(self, input):
        num_channels = K.int_shape(input)[-1]
        
        # Pass re-scaled L2 distance through Tanh function
        L2_distance = K.sum(K.square(input), axis=(1, 2, 3))
        ret_val = K.square(K.tanh((self.scalar * L2_distance + self.bias) / num_channels))
        
        # Return final values (always between 0 and 1)
        return ret_val 
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], ) 
        

# Takes twos layer and combines them based on an tanh activation
def GetTanhCombination(name=None):
    def func(components):
        tanh, x, y = components     # X is cur_term, Y is prev_term

        # Update tanh length    
        len_diff = len(K.int_shape(x)) - len(K.int_shape(tanh))
        for _ in range(len_diff):
            tanh = K.expand_dims(tanh)
            
        # Get linear combo based on tanh value
        inverse_tahn = K.ones_like(y) - tanh
        
        return (x * tanh) + (y * inverse_tahn)
        
    return Lambda(func, name=name)    