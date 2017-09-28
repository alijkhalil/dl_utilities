# Import statements
import os, gc, sys, time
import numpy as np
import math, random

import Queue
import warnings, multiprocessing, threading
from collections import deque

import tensorflow as tf

from dl_utilities.callbacks import callback_utils as cb_utils
from keras import models, metrics
from keras import backend as K

from keras.optimizers import *
from keras.models import Model
from keras.callbacks import Callback


DEFAULT_TRIP_LOSS_PERCENT=0.1



##################   Multi-threaded triplet batch iterator   ##################

class multi_thread_trip_gen(object):
    def __init__(self, queue_size=24, nthreads=8):		
        # Set-up presistent global/shared variables
        self.activity_signal = multiprocessing.Value('i', 0)
        
        self.manager = multiprocessing.Manager()
        self.weight_list = self.manager.list()
        self.weight_lock = multiprocessing.Lock()
        self.weight_callback = cb_utils.UpdateSharedWeights(self.weight_list, self.weight_lock)
                                            
        self.thread_list = []		
        self.q = multiprocessing.Queue(maxsize=queue_size)									
                                            
        self.skip_elements = 50000
        self.skip_values = [ random.random() for _ in xrange(self.skip_elements) ]									
                                            
        self.new_alpha = multiprocessing.Value('d', 0.0)
        self.hardness = multiprocessing.Value('d', 0.0)
        
        self.parameter_list = self.manager.dict()		
        self.any_inside = multiprocessing.Value('i', 0)
        
        
        # Generator initialization helper functions										
        def get_hardness_specific_values(cur_hardness, num_samp, num_cl, 
                                skip_percentage, batch_size, max_per_anchor):			
                                
            min_batches_per_iter = (float(num_samp * num_cl) / batch_size)
            
            # Get empty factor based on level of redundancy
            # High redundancy ==> more batches, higher skip percent, lower empty factor
            if cur_hardness == 0:
                empty_factor = 0.65
                min_batches = int(math.ceil(4 * min_batches_per_iter))
                max_p_anchor = max_per_anchor 
                cur_skip = float(skip_percentage) * 0.5
                
            elif cur_hardness == 1:
                empty_factor = 0.7
                min_batches = int(math.ceil(3 * min_batches_per_iter))
                max_p_anchor = max_per_anchor
                cur_skip = float(skip_percentage) * 0.5
                
            else:
                empty_factor = 0.6
                min_batches = int(math.ceil(5 * min_batches_per_iter))
                max_p_anchor = 1
                cur_skip = float(skip_percentage)
                
            return empty_factor, min_batches, max_p_anchor, cur_skip
        
        
        # Main worker thread function
        def worker(work_sig, qlist, wlist, wlock, synced_alpha, 
                        synced_hardness, param_list, any_inside):
                    
            # Set up envirnoment and session
            # Use the "nvidia-smi" command for more info on GPU memory usage
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            
            pid_val = os.getpid()
            random.seed(pid_val)
            np.random.seed(pid_val)
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(config=config)
            K.set_session(sess)
            
            
            # Main functionality loop
            while True:
                # Regularly check if work should be done
                time.sleep(2 * nthreads)
                with work_sig.get_lock():
                    if work_sig.value == 0:
                        continue

                        
                # If triplet training is ready, set up parameters
                samples, labels = param_list['training_data']
                class_divisions = param_list['class_indices']
                
                scratch_model_func = param_list['new_model_func']
                data_aug = param_list['data_aug']
                
                overall_num_classes = labels.shape[1]
                overall_num_samples = samples.shape[0]
                
                num_cl = int(math.ceil(math.sqrt(overall_num_classes)))
                num_samp = int(math.ceil(math.sqrt(overall_num_samples // overall_num_classes)))
                
                batch_size = param_list['batch_size']
                max_per_anchor = param_list['max_per_anchor']
                skip_percentage = float(param_list['skip_percentage'])
                
                init_alpha = float(param_list['margin'])				
                dynamic_margin = param_list['dynamic_margin']

                is_L2 = param_list['is_L2']
                if is_L2:
                    distance_func = (lambda x: np.sum(np.square(x), 
                                                        axis=-1, keepdims=True))
                else:
                    distance_func = (lambda x: np.sum(np.abs(x), 
                                                        axis=-1, keepdims=True))
                                                        
                # If dynamic alpha, set up a deque over 5 random selections to set "good" margin value 
                if dynamic_margin:				
                    settled = False
                
                    num_iter = int(5 * (1 - skip_percentage) * num_cl * num_samp)
                    loss_list = deque(maxlen=num_iter)			
                    running_additions = float(0.0)
                    
                    min_elements_per_anchor_per_neg_class = num_samp * (num_samp - 1) * 0.3
                    max_elements_per_anchor_per_neg_class = num_samp * (num_samp - 1) * 0.9
                        
                    min_additions = num_iter * min_elements_per_anchor_per_neg_class	
                    max_additions = num_iter * max_elements_per_anchor_per_neg_class

                    
                # Initialize dummy model and function to get final embedding layers
                K.clear_session()				
                tmp_model = scratch_model_func()
                                
                embed_layer = tmp_model.get_layer("final_embeddings")					
                f = K.function([tmp_model.input] + [K.learning_phase()], [embed_layer.output])		
                
                
                # Iterate through samples until enough triplets generated for training
                skip_index = 0
                valid_triplets=[]
                aug_images = {}
                
                while True:
                    final_indices = []
                    embeddings = []
                    
                    # Occasionally garbage collect to avoid bloating system memory
                    gc.collect()
                    
                    # Check if work should continue to be done
                    with work_sig.get_lock():
                        if work_sig.value == 0:
                            break					
                            
                    # Load latest model weights (with a deep model copy)								
                    with wlock:
                        if len(wlist):						
                            tmp_model.set_weights(wlist)
                        else:
                            continue
                            
                    # Increment inside/"still working" counter		
                    with any_inside.get_lock():
                        any_inside.value = (any_inside.value + 1)					
                                                
                                                
                    # Get normalized embeddings from random examples from each of the subset of classes
                    cl_list = random.sample(range(overall_num_classes), num_cl)
                    for i in cl_list:
                        cl_sample_list = random.sample(range(len(class_divisions[i])), num_samp)
                        final_indices.append(np.array(class_divisions[i])[cl_sample_list])
                    
                    for i in range(num_cl):
                        cur_samples = samples[final_indices[i]]
                        
                        # Get augmented images if there is an available generator
                        if data_aug:
                            cur_labels = np.ndarray(shape=(num_samp, 1))
                            cur_labels[:, 0] = np.array(range(num_samp))
                            
                            cur_samples, cur_labels = (data_aug.flow(
                                                        cur_samples, cur_labels,
                                                        batch_size=num_samp).next())
                                                
                            cur_samples = cur_samples[cur_labels[:, 0].astype(int)]
                            
                            # Save them to pass to model for training
                            for tmp_index, real_index in enumerate(final_indices[i]):
                                aug_images[real_index] = cur_samples[tmp_index]
                            
                        preloaded_embeddings = f([cur_samples, 0.])[0]
                        preloaded_embeddings /= np.sqrt(np.sum(np.square(preloaded_embeddings), 
                                                                axis=-1, keepdims=True))
                        
                        embeddings.append(preloaded_embeddings)
                    
                    # Get current hardness values
                    with synced_hardness.get_lock():
                        cur_hardness = int(synced_hardness.value)
                        hardness_remainder = float(synced_hardness.value - cur_hardness)
                        
                    empty_factor, min_batches, max_p_anchor, skip_percent = (
                            get_hardness_specific_values(cur_hardness, num_samp, num_cl, 
                                                            skip_percentage, batch_size, 
                                                            max_per_anchor))
                    med_percent = skip_percent + ((1 - skip_percent) * (1 - hardness_remainder))

                    
                    # Actually begin selecting the triplets
                    for pos_class in range(num_cl):
                        for anchor_index in range(num_samp):					
                            # Randomly skip some anchors to limit redundancy
                            skip_index += 1
                            if skip_index == self.skip_elements:
                                skip_index = 0
                            
                            tmp_hardness = cur_hardness
                            cur_skip = self.skip_values[skip_index]
                            if skip_percent > cur_skip:
                                    continue
                                    
                            elif cur_hardness == 2 and med_percent > cur_skip:
                                cur_skip = int(cur_skip // 0.000001)
                                if (cur_skip & 1):
                                    tmp_hardness = 1
                                else:
                                    tmp_hardness = -1
                                
                            # Get positive distances
                            anchor_embedding = embeddings[pos_class][anchor_index]
                            pos_distances = distance_func(embeddings[pos_class] - anchor_embedding)
                                                    
                            pos_distances[anchor_index] = np.NaN
                            pos_distances = np.tile(pos_distances, num_samp)
                            
                            # Iterate through negative classes
                            final_num_trips = 0
                            indice_trips = []
                                                    
                            poss_neg_classes = [ x for x in range(num_cl) if x != pos_class ]
                            for neg_class in poss_neg_classes:									
                                # Get negative distances
                                neg_distances = distance_func(embeddings[neg_class] - anchor_embedding)
                                neg_distances = np.transpose(neg_distances)															
                                
                                # For each anchor against each negative class, get triplets based on hardness
                                warnings.filterwarnings('ignore')
                                if tmp_hardness < 2:
                                    if tmp_hardness == 0: 
                                        some_triplets = np.where(np.logical_and((neg_distances - pos_distances) > 0, 
                                                                    neg_distances - pos_distances < init_alpha))		
                                    
                                    elif tmp_hardness == 1:
                                        some_triplets = np.where((neg_distances - pos_distances) < init_alpha)
                                    
                                    else:
                                        some_triplets = np.where((neg_distances - pos_distances) < 0)
                                        
                                    # Get number of additional triplets
                                    num_new_trips = some_triplets[0].shape[0]
                                    
                                    # Change margin immediately if queue is filling at undesirable rate
                                    if dynamic_margin:									
                                        if len(loss_list) == num_iter: 
                                            settled = True
                                            running_additions -= loss_list.popleft()

                                            # If not enough additions, means that margin is too small/easy
                                            if min_additions > running_additions:
                                                with synced_alpha.get_lock():
                                                    if init_alpha != synced_alpha.value:
                                                        init_alpha = synced_alpha.value 
                                                    else:
                                                        init_alpha *= 1.25
                                                        synced_alpha.value = init_alpha
                                                        
                                                    print("Enlarging margin to %f... %d" % (init_alpha, os.getpid()))
                                                    sys.stdout.flush()
                                                        
                                                loss_list.clear()
                                                running_additions = float(0.0)
                                                
                                                valid_triplets = []
                                                settled = False
                                                
                                            # If too many additions, means that margin is too large/hard
                                            elif max_additions < running_additions:
                                                with synced_alpha.get_lock():
                                                    if init_alpha != synced_alpha.value:
                                                        init_alpha = synced_alpha.value 
                                                    else:
                                                        init_alpha *= 0.8
                                                        synced_alpha.value = init_alpha
                                                    
                                                    print("Shrinking margin to %f... %d" % (init_alpha, os.getpid()))
                                                    sys.stdout.flush()
        
                                                loss_list.clear()
                                                running_additions = float(0.0)

                                                valid_triplets = []
                                                settled = False
                                        
                                        loss_list.append(num_new_trips)
                                        running_additions += num_new_trips
                                        
                                        # Add valid triplets for the given anchor and negative class
                                        if settled:
                                            for i in range(num_new_trips):
                                                valid_triplets.append((final_indices[pos_class][anchor_index], 
                                                                        final_indices[pos_class][some_triplets[0][i]],
                                                                        final_indices[neg_class][some_triplets[1][i]]))
                                    
                                    # If not dynamic, then limit to "num_new_trips" examples per anchor
                                    else:
                                        for i in range(num_new_trips):
                                            indice_trips.append((neg_class, some_triplets[0][i], some_triplets[1][i]))
                                            
                                        final_num_trips += num_new_trips
                                        if neg_class == poss_neg_classes[-1]:
                                            if max_p_anchor >= final_num_trips:
                                                vals = range(final_num_trips)
                                            else:
                                                vals = random.sample(range(final_num_trips), max_p_anchor)
                                                
                                            for i in vals:
                                                valid_triplets.append((final_indices[pos_class][anchor_index], 
                                                                        final_indices[pos_class][indice_trips[i][1]],
                                                                        final_indices[indice_trips[i][0]][indice_trips[i][2]]))
                                
                                # Get absolute hardest triplet per anchor							
                                else:
                                    min_neg = np.nanargmin(neg_distances[0])
                                    max_pos = np.nanargmax(pos_distances[:, 0])
                                    
                                    # Select triplet only if it is harder than all other examples for anchor
                                    new_dist = neg_distances[0, min_neg] - pos_distances[max_pos, 0]
                                    if (neg_class == poss_neg_classes[0] or best_dist > new_dist): 
                                        best_dist = new_dist
                                        index_vals = (neg_class, max_pos, min_neg)
                                            
                                    # When done, add hardest triplet to the "valid_triplets" list
                                    if neg_class == poss_neg_classes[-1]:
                                        valid_triplets.append((final_indices[pos_class][anchor_index], 
                                                                final_indices[pos_class][index_vals[1]],
                                                                final_indices[index_vals[0]][index_vals[2]]))
                                    
                                    
                    # Check to see whether it is time to shuffle and empty the part of the queue
                    max_size = batch_size * min_batches
                    if (len(valid_triplets) > max_size):
                        np.random.shuffle(valid_triplets)

                        min_leftover = math.ceil(max_size * float(empty_factor))
                        while (len(valid_triplets) > min_leftover):
                            indices = [ valid_triplets.pop(0) for _ in range(batch_size) ]

                            final_indices = []
                            for (x, y, z) in indices:
                                final_indices.append(x)
                                final_indices.append(y)
                                final_indices.append(z)
                            
                            # Add either augmented or original images/labels to the queue
                            if data_aug:
                                trip_images = np.ndarray(shape=((len(final_indices), ) + samples.shape[1:]))
                                for tmp_index, aug_index in enumerate(final_indices):
                                    trip_images[tmp_index] = aug_images[aug_index]
                                
                                trip_labels = labels[final_indices]
                            
                            else:
                                trip_images = samples[final_indices]
                                trip_labels = labels[final_indices]

                            qlist.put(([trip_images], [trip_labels, trip_labels]))

                            
                    # Decrement inside counter
                    with any_inside.get_lock():
                        any_inside.value = (any_inside.value - 1)
                            
                    # Ensure embeddings are removed from the system
                    del final_indices
                    del cur_samples
                    del embeddings

                
        # Start "nthreads" processes to concurrently yield triplets
        for _ in range(nthreads):
            t = multiprocessing.Process(target=worker, args=(
                                                self.activity_signal, self.q, 
                                                self.weight_list, self.weight_lock, 
                                                self.new_alpha, self.hardness, 
                                                self.parameter_list, self.any_inside))
            t.daemon = True
            
            self.thread_list.append(t)
            t.start()
        
        
    # Iterator returns itself
    def __iter__(self):
        return self

        
    # Python 2 and 3 compatibility for the standard iter "next" call
    def __next__(self):
        return self.next()

        
    def next(self):			
        return self.q.get()


    # Get margin value after dynamic alterations
    def get_new_alpha(self):
        with self.new_alpha.get_lock():
            return self.new_alpha.value


    # Set new hardness as model becomes more tailored to triplet loss
    def set_new_hardness_value(self, new_hardness):
        with self.hardness.get_lock():
            self.hardness.value = new_hardness


    # Check for readiness
    def is_ready_for_start(self):
        with self.activity_signal.get_lock():
            if self.activity_signal.value != 0:
                return False
            
        with self.any_inside.get_lock():
            if self.any_inside.value != 0:
                return False	

        return True


    # Function to start the generator up (either initially or after it was stopped) 
    def start_activity(self, training_data, class_indices, 
                        scratch_model_func, batch_size, margin, 
                        dynamic_margin=True, is_L2=True, 
                        data_aug=None, max_per_anchor=5, hardness=2, 
                        skip_percentage=0.3):
                
        # Check to ensure that there is no residuals activity from last call to workers	
        if not self.is_ready_for_start():
            return False
        
        
        # Save parameters		
        self.parameter_list['training_data'] = (np.array(training_data[0]), 
                                                    np.array(training_data[1]))
        self.parameter_list['class_indices'] = class_indices
        
        self.parameter_list['new_model_func'] = scratch_model_func
        self.parameter_list['data_aug'] = data_aug
        
        self.parameter_list['batch_size'] = batch_size
        self.parameter_list['margin'] = margin
        self.parameter_list['dynamic_margin'] = dynamic_margin
        
        self.parameter_list['skip_percentage'] = skip_percentage
        self.parameter_list['max_per_anchor'] = max_per_anchor
        self.parameter_list['is_L2'] = is_L2
                
        with self.new_alpha.get_lock():
            self.new_alpha.value = margin
            
        with self.hardness.get_lock():
            self.hardness.value = hardness
            
        # Start workers
        with self.activity_signal.get_lock():
            self.activity_signal.value = 1		
        
        # Return success
        return True
            
            
    # Stop the generator from continuing to add to the queue
    def stop_activity(self):
        with self.activity_signal.get_lock():
            self.activity_signal.value = 0
        
        # Bleed the queue to ensure that no threads are still working
        i = 0
        while True:
            if i == 10:
                i = 0
                with self.any_inside.get_lock():
                    sys.stdout.flush()
                    if self.any_inside.value == 0:
                        break
            
            try:
                self.q.get_nowait()
            except Queue.Empty:
                pass
            
            i += 1

        # Empty weights
        while len(self.weight_list):
            self.weight_list.pop()


    # Stop the generator altogether (e.g. destructor) at the end of main thread
    def stop_all_threads(self):
        for thread in self.thread_list:
            if thread.is_alive():
                thread.terminate()
            
        self.q.close()			

    
    

#####################   Loss functions   #####################

# Triplet accuracy (only looking at anchors)
def trip_accuracy(y_true, y_pred):
    # Seperate embeddings into the triplets
    num_classes = y_pred._keras_shape[-1]

    trip_pred = K.reshape(y_pred, (-1, 3, num_classes))
    trip_labels = K.reshape(y_true, (-1, 3, num_classes))

    # Return correct classification 
    return metrics.categorical_accuracy(trip_labels[:,0], trip_pred[:,0])

 
# Triplet cross-entropy loss function (only looking at anchors)
def trip_x_entropy_loss(y_true, y_pred):
    # Seperate embeddings into the triplets
    num_classes = y_pred._keras_shape[-1]

    trip_pred = K.reshape(y_pred, (-1, 3, num_classes))
    trip_labels = K.reshape(y_true, (-1, 3, num_classes))

    # Return cross-entropy without positive and negative examples (e.g. anchor only)
    return K.mean(K.categorical_crossentropy(trip_pred[:,0], trip_labels[:,0]))


# Triplet L1 loss function	
def trip_l1_loss(margin):
    def l1_helper(y_true, y_pred):
        # Seperate embeddings
        embed_size = y_pred._keras_shape[-1]
        embeddings = K.reshape(y_pred, (-1, 3, embed_size))

        # Optionally normalize each embedding here
        norm_achor = (embeddings[:,0] / 
                        K.sqrt(K.sum(K.square(embeddings[:,0]), axis=-1, keepdims=True))) 
        norm_pos = (embeddings[:,1] / 
                        K.sqrt(K.sum(K.square(embeddings[:,1]), axis=-1, keepdims=True))) 
        norm_neg = (embeddings[:,2] / 
                        K.sqrt(K.sum(K.square(embeddings[:,2]), axis=-1, keepdims=True)))
                        
        # Get difference
        positive_distance = K.sum(K.abs(norm_achor - norm_pos), axis=-1)
        negative_distance = K.sum(K.abs(norm_achor - norm_neg), axis=-1)
        return K.mean(K.maximum(0.0, margin + positive_distance - negative_distance))

    return l1_helper


# Triplet L2 loss function		
def trip_l2_loss(margin):
    def l2_helper(y_true, y_pred):
        # Seperate embeddings
        embed_size = y_pred._keras_shape[-1]
        embeddings = K.reshape(y_pred, (-1, 3, embed_size))

        # Optionally normalize each embedding here
        norm_achor = (embeddings[:,0] / 
                        K.sqrt(K.sum(K.square(embeddings[:,0]), axis=-1, keepdims=True))) 
        norm_pos = (embeddings[:,1] / 
                        K.sqrt(K.sum(K.square(embeddings[:,1]), axis=-1, keepdims=True))) 
        norm_neg = (embeddings[:,2] / 
                        K.sqrt(K.sum(K.square(embeddings[:,2]), axis=-1, keepdims=True))) 
        
        # Get difference
        positive_distance = K.sum(K.square(norm_achor - norm_pos), axis=-1)
        negative_distance = K.sum(K.square(norm_achor - norm_neg), axis=-1)
        return K.mean(K.maximum(0.0, margin + positive_distance - negative_distance))

    return l2_helper




#####################   General Utilities   #####################

# Get class indices to be ready for generator
def break_down_class_indices(training_labels):
    class_indices = [[] for i in range(training_labels.shape[1])]
    for index, one_hot_label in enumerate(training_labels):
        class_indices[np.argmax(one_hot_label)].append(index)

    return class_indices 


# Return compiled triplet version of a model
def convert_model_to_trip_model(model, opt, margin, expected_loss, 
                                percent_trip_loss=DEFAULT_TRIP_LOSS_PERCENT, 
                                is_L2=True):
    try:
        # Initialize model
        final_embeddings = model.get_layer("final_embeddings").output
        predictions = model.get_layer("predictions").output
        
        trip_model = Model(inputs=model.input, outputs=[final_embeddings, predictions])
        
        # Set loss ratios
        if is_L2:
            trip_loss_func = trip_l2_loss(margin=margin)
            ideal_final_margin_portion = 0.25
            
        else:
            trip_loss_func = trip_l1_loss(margin=margin)			
            ideal_final_margin_portion = 0.5
        
        # Get normal and triplet margin loss weight
        margin_factor = float((percent_trip_loss * expected_loss) 
                                / (margin * ideal_final_margin_portion))
        
        xtropy_factor = 1.0 - float(percent_trip_loss)
        
        # Compile model
        print(margin_factor, xtropy_factor)
        trip_model.compile(optimizer=opt, 
                        loss={'final_embeddings': trip_loss_func, 
                            'predictions': trip_x_entropy_loss}, 
                        metrics={'predictions': trip_accuracy},
                        loss_weights={'final_embeddings': margin_factor, 
                            'predictions': xtropy_factor})
        
        # Return embeddings
        return trip_model
        
    except ValueError:
        # Return error if expected model layer not found
        print("The model provided to the function did not have a layer " +
                "labelled 'final_embeddings' and/or 'predictions'.")
                
        return None


# Function for heavy lifting of triplet training (returns history dictionary)		
def train_triplet_model(model, opt, trip_worker, scratch_model_func, 
                        train_data, valid_data, margin, expected_loss, batch_size, 
                        epochs, step_size=5, num_easy_steps=2, num_medium_steps=2, 
                        percent_trip_loss=DEFAULT_TRIP_LOSS_PERCENT, dynamic_margin=False, 
                        warm_up_examples_per_anchor=3, is_L2=True, image_aug=None, 
                        callbacks=[]):	
        
    # Do checks to ensure agruments are valid
    assert epochs % step_size == 0, \
                    ("Number of epochs should be divisible by %d." % step_size)

    assert (((num_easy_steps + num_medium_steps + 1) * step_size) <= epochs), \
                    ("Not enough training epochs. Need at least one 'step_size' epochs " \
                       "after the easy/medium training.")

                       
    # Get classes based on training data
    train_images, train_labels = train_data
    class_indices = break_down_class_indices(train_labels)
        
        
    # Start generator for triplet selections
    if not trip_worker.is_ready_for_start():
        trip_worker.stop_activity()
        
    trip_worker.start_activity(train_data, class_indices, scratch_model_func,
                                    batch_size, margin, dynamic_margin, is_L2, 
                                    image_aug, warm_up_examples_per_anchor, 
                                    hardness=2.99, skip_percentage=0.4)
        

    # Add weight monitoring callback to current list
    callbacks.append(trip_worker.weight_callback)


    # Model specifically for printing evaluation metrics
    eval_model = Model(inputs=model.input, outputs=model.get_layer("predictions").output)
    eval_model.compile(optimizer='adam', 
                        loss='categorical_crossentropy', 
                        metrics =['accuracy', metrics.top_k_categorical_accuracy])
        

    # Begin sequence for triplet training	
    init_lr = float(K.get_value(model.optimizer.lr))

    anything_changed = False
    overall_history = {}

    easy_iters = num_easy_steps * step_size
    medium_iters = easy_iters + (num_medium_steps * step_size)
    hard_iters = epochs - medium_iters
    num_steps_before_clear = 6

    for iter in range(0, epochs, step_size):
        # Garbage collect to avoid exhuasting system memory
        gc.collect()
        
        
        # Set hardness
        if iter < easy_iters:
            hardness = 0.0
            multiply_factor = 1.85
            
        elif iter < medium_iters:
            hardness = 1.0
            multiply_factor = 1.7
            
        else:
            hardness = 2.99
            if (hard_iters // step_size) > 3:
                progress_remainder = math.floor((4.0 * 
                                            float(hard_iters - iter - step_size + medium_iters)) 
                                            / hard_iters)
                hardness -= (progress_remainder * 0.2)
            
            multiply_factor = 1.55
        
        trip_worker.set_new_hardness_value(hardness)

        
        # Actually perform training		
        hist = model.fit_generator(
                    trip_worker, 
                    steps_per_epoch=
                        (train_images.shape[0] // (batch_size * multiply_factor)),
                    epochs=(iter + step_size),
                    initial_epoch=iter,
                    callbacks=callbacks)
        
        
        # Check if margin was changed at all by generator
        next_iter = iter + step_size
        if dynamic_margin:
            new_margin = trip_worker.get_new_alpha()
            if new_margin != margin:
                anything_changed = True
                print("Changing margin from %d to %d!" % (margin, new_margin))
                
                if epochs != next_iter:
                    margin = new_margin
                    tmp_lr = cb_utils.get_cosine_scaler(init_lr, next_iter, epochs)	
                
                    model = convert_model_to_trip_model(
                                                model, SGD(lr=tmp_lr), margin, 
                                                expected_loss, percent_trip_loss,
                                                is_L2)
        
        
        # Get progress report
        metric_names = eval_model.metrics_names
        metric_vals = eval_model.evaluate(valid_data[0], valid_data[1], verbose=0)
        
        print(eval_model.metrics_names)
        print(metric_vals)
        print("\n")
        
        # Accumulate full history
        if iter == 0:
            for key, val in zip(metric_names, metric_vals):
                hist.history["val_" + key] = [ val ] * 2
            
            overall_history = hist.history
            
        else:
            for key, val in zip(metric_names, metric_vals):
                overall_history["val_" + key].extend([ val ] * 2)
        
            for key in hist.history:
                overall_history[key].extend(hist.history[key])
                
                
        # Clear session every "num_steps_before_clear" iterations
        if next_iter % (step_size * num_steps_before_clear) == 0:
            anything_changed = True
            print("Clearing session!")

            tmp_weights = model.get_weights()
            K.clear_session()

            # Recreate model
            model = scratch_model_func()
            model.set_weights(tmp_weights)
            
            model.layers[-1].name = "predictions"
            tmp_lr = cb_utils.get_cosine_scaler(init_lr, next_iter, epochs)	
                        
            model = convert_model_to_trip_model(model, SGD(lr=tmp_lr), margin, 
                                                    expected_loss, percent_trip_loss, 
                                                    is_L2)

        # Recreate evaluation model if there is a new model
        if anything_changed and epochs != next_iter:
            anything_changed = False
            
            eval_model = Model(inputs=model.input, outputs=model.get_layer("predictions").output)
            eval_model.compile(optimizer='adam', 
                                loss='categorical_crossentropy', 
                                metrics =['accuracy', metrics.top_k_categorical_accuracy])
                        

    # Stop generator activity
    print("Bleeding the remaining elements from the triplet queue!")
    trip_worker.stop_activity()


    # Return accumulated history dictionary
    return overall_history, model
