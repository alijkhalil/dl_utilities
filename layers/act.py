from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings, math

import keras.backend as K
from keras.layers.core import Lambda



''' 
These are custom layers for use in an ACT cell.

Their names are geared towards use in an ACT cell 
and therefore, may not describe their functionality 
in the most general way.
'''

# Essentially a wrapper for a switch gate
def FlagLayer(init_var, name=None):
    def func(tensors, init_var=init_var):
        return K.switch(init_var, tensors[0], tensors[1])
    
    return Lambda(func, name=name)    


# Create a zero-ed layer with a custom shape    
def CreateCustomShapeLayer(final_dim, name=None):
    def func(x, final_dim=final_dim):
        initial_state = K.zeros_like(x)  # (samples, input_dim)
        initial_state = K.sum(initial_state, axis=-1)  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        
        # Build zero-ed intermediate states by getting dimension of each state        
        final_state = K.tile(initial_state, [1, final_dim])  # (samples, final_dim)
        return final_state
        
    return Lambda(func, name=name)    
    
    
# Wrapper for a "zeros" call
def ResetLayer(name=None):
    def func(x):
        return K.zeros_like(x, dtype='float32')
        
    return Lambda(func, name=name)     


# Converts tensor to float 0's and 1's based on comparison with another tensor
def CompLayer(name=None):
    def func(tensors):
        return K.cast(K.less(tensors[0], tensors[1]), dtype='float32')    
    
    return Lambda(func, name=name)     


# Takes layer (of presumably probabilities) and gets one minus that layer
def OneMinusLayer(name=None):
    def func(x):
        return K.ones_like(x) - x
        
    return Lambda(func, name=name)


# Multiplies layer by a scalar    
def MultiplyByScalar(value, name=None):
    def func(x, value=value):
        return value * x
        
    return Lambda(func, name=name)     
    

# Sets a layer to an arbitrary, constant value (essentially an equal operator)
def SetterLayer(value, name=None):
    def func(x, value=value):
        return K.zeros_like(x, dtype='float32') + value
        
    return Lambda(func, name=name)     