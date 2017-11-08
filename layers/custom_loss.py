# Import statements
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import keras.backend as K
from keras.layers import Lambda



'''
Custom loss layers -

Intended for use in situations where the loss function is unconventional.

More specifically, it helps in situations where values aside from 
    'y_true' and 'y_pred' are required.

To use such a layer, you should simply add it to a model as its last layer.
Then, you can can call compile so that its loss function returns the final layer.

Example code:
    input = Input((input_dim,))
    last_layer1, last_layer2 = example_model(input)
    loss_val = CustomLossLayer([last_layer1, last_layer2])

    final_model = Model(input, loss_val)
    final_model.compile(loss=(lambda y_true, y_pred: y_pred), optimizer='adam')
'''

# Normal cross-entropy 		
def CalculateNormalXEntropyLoss(name=None):
    def func(tensors):
        # Seperate tensor components
        true_output, prediction = tensors
        
        # Normalize outputs
        prediction /= K.sum(prediction,
                                axis=len(prediction.get_shape()) - 1,
                                keepdims=True)
                                
        # Clip them to ensure that they are between [EPSILON, 1. - EPSILON]                        
        prediction = K.clip(prediction, K.common._EPSILON, 1. - K.common._EPSILON)
        
        # Get cross entropy value
        xentropy_component = - K.sum(true_output * K.log(prediction),
                                        axis=len(prediction.get_shape()) - 1)
        
        # Return it
        return K.expand_dims(xentropy_component, 1)		
                    
    return Lambda(func, name=name)

    
# ACT loss function (featuring a ponder cost component)
def CalculateACTLoss(name=None):
    def func(tensors):
        # Seperate tensor components
        true_output, prediction, counter_final, remainder_final = tensors
        
        # Normalize outputs
        prediction /= K.sum(prediction,
                                axis=len(prediction.get_shape()) - 1,
                                keepdims=True)
                                
        # Clip them to ensure that they are between [EPSILON, 1. - EPSILON]                        
        prediction = K.clip(prediction, K.common._EPSILON, 1. - K.common._EPSILON)
        
        # Get cross entropy value
        xentropy_component = - K.sum(true_output * K.log(prediction),
                                        axis=len(prediction.get_shape()) - 1)
        
        # Get overall ponder cost
        ponder_cost = (xentropy_component + K.sum(counter_final) + K.sum(remainder_final))
                    
        # Return it
        return K.expand_dims(ponder_cost, 1)
                    
    return Lambda(func, name=name) 	