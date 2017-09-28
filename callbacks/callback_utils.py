from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings, math

import numpy as np
import tensorflow as tf

import keras
from keras.models import Model
from keras.callbacks import Callback, LearningRateScheduler

import keras.backend as K
from keras.optimizers import *



# Cosine annealing function for decreasing rate of change in a variable
def get_cosine_scaler(base_val, cur_iter, total_iter):
    if cur_iter < total_iter:
        return (0.5 * base_val * (math.cos(math.pi * 
                        (cur_iter % total_iter) / total_iter) + 1))
    else:
        return 0

        
# Callback for LR Scheduler based on cosine annealing function        
def CosineLRScheduler(init_lr, total_epochs):
    def variable_epochs_cos_scheduler(cur_epoch):
        return get_cosine_scaler(init_lr, cur_epoch, total_epochs)
        
    return LearningRateScheduler(variable_epochs_cos_scheduler)

                
# Dropout callback for progressively increasing "final_dropout" layer dropout 
class DynamicDropoutWeights(Callback):
    def __init__(self, final_dropout):
        super(DynamicDropoutWeights, self).__init__()

        if final_dropout < 0.3:
            range_val = final_dropout * 0.375
        elif final_dropout < 0.6:
            range_val = 0.175
        else:
            range_val = 0.25
         
        self.final_dropout = final_dropout 
        self.range = range_val
        
    def on_epoch_begin(self, epoch, logs={}):
        # At start of every epoch, slowly increase dropout towards final value                                                        
        total_epoch = self.params["epochs"]
        subtract_val = get_cosine_scaler(self.range, (epoch + 1), total_epoch)            

        dropout_layer = self.model.get_layer("final_dropout")            
        dropout_layer.rate = (self.final_dropout - subtract_val)
        
        
# Drop Path (e.g. Stochastic Depth) Callback and associated functions
def set_up_death_rates(cur_dt, desired_death_rate, gates_p_layer):
    # Convert 'gates_p_layer' list into running sum
    orig = np.array(gates_p_layer)
    new_running_sum = np.ndarray(shape=(len(gates_p_layer)))
    for i in range(len(orig)):
        new_running_sum[i] = np.sum(orig[:i+1])
        
    # Get layer of gate and use it to calculate death_rate for layer
    cur_sum_index = 0
    total = new_running_sum[cur_sum_index]

    cur_layer = 1
    num_layers = len(gates_p_layer)
    for i, tb in enumerate(cur_dt):
        if i >= total:
            cur_layer += 1
            cur_sum_index += 1
            
            total = new_running_sum[cur_sum_index]

        portion_of_dr = float(cur_layer) / num_layers
        K.set_value(tb["death_rate"], (desired_death_rate * portion_of_dr))

        
class DynamicDropPathGates(Callback):
    def __init__(self, final_death_rate, cur_dt, gates_p_layer, update_freq=1):
        super(DynamicDropPathGates, self).__init__()
        
        self.dt = cur_dt
        self.gates_p_layer = gates_p_layer
        self.update_freq = update_freq
        
        if final_death_rate < 0.3:
            range_val = final_death_rate * 0.375
        elif final_death_rate < 0.6:
            range_val = 0.175
        else:
            range_val = 0.25

        self.final_death_rate = final_death_rate
        self.range = range_val

    def on_epoch_begin(self, epoch, logs={}):
        # Update death rates
        total_epoch = self.params["epochs"]
        subtract_val = get_cosine_scaler(self.range, (epoch + 1), total_epoch)
        cur_death_rate = (self.final_death_rate - subtract_val)             
        
        set_up_death_rates(self.dt, cur_death_rate, self.gates_p_layer)
        
    def on_batch_begin(self, batch, logs={}):
        # Randomly close some gates at start of every 'update_freq' batch
        if batch % self.update_freq == 0:
            rands = np.random.uniform(size=len(self.dt))
            for tb, rand in zip(self.dt, rands):
                if rand < K.get_value(tb["death_rate"]):
                    K.set_value(tb["gate"], 0)

    def on_batch_end(self, batch, logs={}):
        # Re-open all gates at the end of every 'update_freq' batches
        total_steps = (self.params["steps"] - 1)
        if (batch % self.update_freq == 0 or batch >= total_steps):
            for tb in self.dt:
                K.set_value(tb["gate"], 1)
                
                
# Callback for triplet loss training to get weights of shared model across processes
class UpdateSharedWeights(Callback):
    def __init__(self, init_weights, wlock):
        super(UpdateSharedWeights, self).__init__()
        self.wlist = init_weights
        self.wlock = wlock
    
    def on_train_begin(self, logs={}):
        with self.wlock:
            if len(self.wlist) == 0:
                for layer_weights in self.model.get_weights():
                    self.wlist.append(layer_weights)

    def on_batch_end(self, batch, logs={}):				
        if batch != 0 and batch % 25 == 0:
            with self.wlock:
                for i, layer_weights in enumerate(self.model.get_weights()):
                    self.wlist[i] = layer_weights
