# Import statements
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings, math
import tensorflow as tf

import keras.backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.layers import Dense, Reshape
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import add, multiply, _Merge
from keras.initializers import Constant


DEFAULT_EPSILON=1E-6



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
        

# Similar layer to "Add" layer (except obviously substracts rather than 'adds')
class Subtract(_Merge):
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output -= inputs[i]
        return output
        
        
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
    

# Layer to get a single spatial embedding from layer of CNN
def GetSpecificSpatialFeatures(w, h, name=None):
    def func(cnn_embed, w=w, h=h):            
        return cnn_embed[:, w, h, :]
        
    return Lambda(func, name=name)

    
# Layer to add positional channels on top of regular RGB (for relational processing)    
def AddPositionalChannels(name=None):
    def func(x):
        # Get key variables
        _, height, width, _ = K.int_shape(x)
        
        height_inc = float(1.0 / height)
        width_inc = float(1.0 / width)
        
        
        # Get y-axis channel
        scalar_val = 0
        height_tensors = []
        for _ in range(height):
            tmp_tensor = K.zeros_like(x)  # (height, width, channels)
            tmp_tensor = K.sum(K.sum(tmp_tensor, axis=-1), axis=1)  # (width,)
            tmp_tensor = K.expand_dims(tmp_tensor, 1)  # (1, width)
            
            scalar_val += height_inc
            tmp_tensor += scalar_val
            
            height_tensors.append(tmp_tensor)
        
        y_axis_channel = K.expand_dims(K.concatenate(height_tensors, axis=1))

        
        # Get x-axis channel
        scalar_val = 0
        width_tensors = []
        for _ in range(width):
            tmp_tensor = K.zeros_like(x)  # (height, width, channels)
            tmp_tensor = K.sum(K.sum(tmp_tensor, axis=-1), axis=-1)  # (height,)
            tmp_tensor = K.expand_dims(tmp_tensor)  # (1, height)

            scalar_val += width_inc
            tmp_tensor += scalar_val
            
            width_tensors.append(tmp_tensor)

        x_axis_channel = K.expand_dims(K.concatenate(width_tensors, axis=-1))

        
        # Return all axises stacked together     
        all_channels = [ x ]
        all_channels.append(y_axis_channel)
        all_channels.append(x_axis_channel)
        
        return K.concatenate(all_channels)
        
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

    
# Layers to return a squeeze/excitation vector or block output 
# These are capable of being integrated into any CNN and usually result in an improved model 
# Based on: https://arxiv.org/pdf/1709.01507.pdf    
#
# WARNING: Not a real layer object so it cannot be added directly to an RNN 
def se_block(reduction_ratio=16, num_final_layers=None, only_excite_vec=False):
    def func(input):        
        # Get original number of filters
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        
        orig_filters = K.int_shape(input)[channel_axis]
        if num_final_layers is None:
            final_filters = orig_filters
        else:
            final_filters = num_final_layers
        
        # Perform spatial dimension reduction
        squeeze = GlobalAveragePooling2D()(input)
        squeeze = Reshape((1, 1, orig_filters))(squeeze)
        
        # Pass through Dense, ReLU, Dense, Sigmoid sequence (for attention vector)
        excite = Dense(int(final_filters // reduction_ratio), activation='relu', 
                        kernel_initializer='he_normal', use_bias=False)(squeeze)
        excite = Dense(final_filters, activation='sigmoid', kernel_initializer='he_normal', 
                        use_bias=False)(excite)

        # Multiply each spatial channel by the attention value
        if only_excite_vec:
            return excite
        else:    
            return multiply([input, excite])
    
    return func
    
    
# Custom swish activation (acting on activations per-channel)
# Based on following papers: https://arxiv.org/pdf/1801.07145.pdf
#                            https://arxiv.org/pdf/1710.05941.pdf
#
# NOTE: Almost a mix of dumbed=down SE block with the derivative effect of a ReLU
class custom_swish(Layer):
    def __init__(self, init_multupler=1.0, **kwargs):
        self.init_multupler = init_multupler

        super(custom_swish, self).__init__(**kwargs)

    def build(self, input_shape):
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
        # Create trainable scalar and bias variables for the layer.
        self.scalar = self.add_weight(name='multipler',
                                      shape=(input_shape[channel_axis], ),
                                      initializer=Constant(value=self.init_multupler),
                                      trainable=True)
                                      
        self.bias = self.add_weight(name='bias_term',
                                      shape=(input_shape[channel_axis], ),
                                      initializer='zeros',
                                      trainable=True)
                                      
        super(custom_swish, self).build(input_shape)

    def call(self, input):
        # Pass re-scaled L2 distance through Tanh function
        ret_val = (input * K.sigmoid((self.scalar * input) + self.bias))
        
        # Return final values (always between 0 and 1)
        return ret_val

    def compute_output_shape(self, input_shape):
        return input_shape
