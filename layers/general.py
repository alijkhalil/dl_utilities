# Import statements
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings, math

import keras.backend as K
from keras.engine.topology import Layer
from keras.layers.core import Lambda
from keras.layers.merge import add



DEFAULT_EPSILON=1E-5



# Layer Normalization (with and without standardization)
class LN(Layer):
    def __init__(self, epsilon=DEFAULT_EPSILON, use_variance=True, **kwargs):
        self.epsilon = epsilon
        self.use_variance = use_variance
        
        super(LN, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.scalar = self.add_weight(name='multipler', 
                                      shape=(1,),
                                      initializer='ones',
                                      trainable=True)
         
        self.bias = self.add_weight(name='bias_term', 
                                      shape=(1,),
                                      initializer='zeros',
                                      trainable=True)
                                      
        super(LN, self).build(input_shape)

    def call(self, input):
        m = K.mean(input, [1], keepdims=True)
        
        if self.use_variance:
            v = K.var(input, [1], keepdims=True)
            normalised_input = (input - m) / K.sqrt(v + self.epsilon)        
        else:
            normalised_input = (input - m) 
            
        ret_val = (self.scalar * normalised_input) + self.bias
        
        return ret_val 
        
    def compute_output_shape(self, input_shape):
        return input_shape 
        

# Helper for getting a subsection of a layer
def crop(start, end=0, dimension=-1, name=None):
    def func(x, start=start, end=end, dimension=dimension):
        if dimension < 0:
            dimension += len(K.int_shape(x))

        if not end:
            end = K.int_shape(x)[dimension]
            
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
			
    return Lambda(func, name=name)
    

# Layer for scaling down activations at test time
def scale_activations(drop_rate, name=None):
    def func(x, drop_rate=drop_rate):
        scale = K.ones_like(x) - drop_rate
        return K.in_test_phase(scale * x, x)
    
    return Lambda(func, name=name)


# Layer for dropping path based on a "gate" variable
#   Input should be formatted: [drop_path, normal_path] 
#   "Gate" variable should be set to 1 to return "normal_path"
def drop_path(gate, name=None):
    def func(tensors, gate=gate):
        return K.switch(gate, tensors[1], tensors[0])
			
    return Lambda(func, name=name)   
        
        
# Drop-in replacement for add function (with drop path functionality incorporated)
def res_add(drop_dict=None):
    if drop_dict is not None:
        def func(tensors, drop_dict=drop_dict):
            # Get death_rate and drop gate variables from table
            gate = drop_dict["gate"]
            death_rate = drop_dict["death_rate"]

            # Get main and scaled (during test time) residual channels
            main_channels, res_channels = tensors
            res_scaled = scale_activations(death_rate)(res_channels)

            # Add scaled value only if gate is open, otherwise keep untouched
            non_drop_path = add([main_channels, res_scaled])      
            ret_val = drop_path(gate)([main_channels, non_drop_path])
        
            return ret_val
            
        ret_fn = func
        
    else:
        ret_fn = add
        
    
    # Return correct function for adding layers
    return ret_fn