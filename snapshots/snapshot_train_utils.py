# Import statements
from __future__ import print_function

import sys
sys.setrecursionlimit(10000)

from dl_utilities.trip_loss import trip_utils
from dl_utilities.callbacks import callback_utils as cb_utils

import numpy as np
import cPickle as pickle
import itertools, os, math

from shutil import copyfile
from keras import metrics

from keras.optimizers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint



# Global variables (functioning primarily as placeholders)
trip_l1_margin = 0.4
trip_l2_margin = 0.3

expected_loss_val = 1.25		
trip_loss_percent = 0.01


# Hyper-parameter strings (for different Snapshots)
COSINE_SGD_OPT = 'sgd'
ADAM_OPT = 'adam'
NADAM_OPT = 'nadam'
ADADELTA_OPT = 'adelta'
TRIP_L2_LOSS = 'triptwo'
TRIP_L1_LOSS = 'tripone'

all_hyperparameters = [ COSINE_SGD_OPT, ADAM_OPT, NADAM_OPT,
							ADADELTA_OPT, TRIP_L1_LOSS, TRIP_L2_LOSS ]

                            
							
####### Helper functions			

# Produces a string representing a series of snapshot models
def get_checkpoint_base_string(seqs, init_lrs, iters, restart_indices):
	base_string = ""
	
	# Set up restart index
	if not isinstance(restart_indices, (list, tuple)):
		restart_indices = [0]
	elif not 0 in restart_indices:
		restart_indices.append(0)
	
	# Checkpoint string just concatenation of sequence strings
	i = 0
	for num_iter, seq, lr in zip(iters, seqs, init_lrs):
		if i in restart_indices:
			base_string += 'r'
		else:
			base_string += 'c'
			
		base_string += (str(num_iter) + seq + str(int(lr * 1000)))
		base_string += "_"
		i += 1
	
	# Return final string
	return base_string[:-1]
	

# Breaks "base" string down into lists representing attributes of each snapshot	
def get_seq_info_from_base_string(base_string):	
	restart_indices = []
	iters = []
	seqs = []
	init_lrs = []
	
	# Break down base string into its components
	tokens = base_string.split("_")
	for i, token in enumerate(tokens):
		# Get restart character
		if token[0] == 'r':
			restart_indices.append(i)
		
		# Get number of iterations
		for m in range(2, len(token)):
			potential_iter = token[1:m]
			if not potential_iter.isdigit():
				final_iter = int(potential_iter[:-1])
				seq_start_index = m - 1
				break
		
		iters.append(final_iter)
		
		# Get seq and LR at the same time
		for m in range(1, len(token) - seq_start_index + 1):
			potential_lr = token[(m * -1):]
			if not potential_lr.isdigit():
				final_lr = float(potential_lr[1:]) / 1000
				seq_end_index = m + 2
				break
		
		seqs.append(token[seq_start_index:seq_end_index])
		init_lrs.append(final_lr)

	# Return break down lists	
	return seqs, init_lrs, iters, restart_indices    
    
    
# Checks if training done a particular model of a Snapshot ensemble		
def training_already_done(subsequence, init_lrs, iter_epochs, 
							restart_indices, cur_iter, weight_dirname):
	
	# Set important strings
	subsequence_string = get_checkpoint_base_string(subsequence, init_lrs, 
													iter_epochs, restart_indices)
	required_ending = (".%d.hdf5" % cur_iter)
	
	# Determine if there's a weight file with matching subsequence 
	for filename in os.listdir(weight_dirname):
		if (filename.startswith(subsequence_string) 
				and filename.endswith(required_ending)):
			
			# Return weight file with overlap
			return filename	
	
	
	# Return None (if there no past training with an overlap)
	return None
    

    
####### Regular training functions	

# For training a normal/simple model with an optimizer
def compile_and_train_single_model(model, opt, train_data, test_data, 
                                nb_epochs, batch_size, history_filepath=None, 
                                aug_gen=None, callbacks=None, **kwargs):			
						
    # Ensure only one output
    new_model = Model(inputs=model.input, outputs=model.layers[-1].output)

    # Compile model
    new_model.compile(loss='categorical_crossentropy',
                    optimizer=opt, 
                    metrics=['accuracy', metrics.top_k_categorical_accuracy])

    # Seperate training data and labels				
    train_images, train_label = train_data

    # Train for specified number of epochs
    if aug_gen:
        hist = new_model.fit_generator(
                    aug_gen.flow(train_images, train_label, batch_size=batch_size),
                    steps_per_epoch=(train_images.shape[0] // batch_size),
                    epochs=nb_epochs, 
                    initial_epoch=0,
                    callbacks=callbacks,
                    validation_data=test_data)
    else:
        hist = new_model.fit(train_images, train_label,
                    batch_size=batch_size,
                    epochs=nb_epochs,
                    initial_epoch=0,
                    callbacks=callbacks,
                    validation_data=(test_images, test_label),
                    shuffle=True)

    # Print evaluation metrics
    print(history_filepath)
    print(new_model.metrics_names)
    print(new_model.evaluate(train_images, train_label, verbose=0))

    # Save metrics to a history file
    if history_filepath:
        pickle.dump(hist.history, open(history_filepath, "wb"))


    # Return history
    return hist.history, new_model


# Main triplet loss training function (mirroring "compile_and_train_single_model" above)
def compile_and_train_trip_model_helper(model, opt, train_data, test_data, nb_epochs, 
                            batch_size, scratch_model_func, trip_worker, use_L2_loss, expected_loss, 
                            margin, trip_loss_portion, step_size, easy_steps, medium_steps,
                            dynamic_margin, warm_up_examples_per_anchor,
                            history_filepath=None, aug_gen=None, callbacks=None):		
						
    # Change name of model's last layers (in case not already the case)
    model.layers[-2].name = "final_embeddings"
    model.layers[-1].name = "predictions"


    # Convert original model for training on a triplet loss
    trip_model = trip_utils.convert_model_to_trip_model(
                                                model, opt, margin, 
                                                expected_loss, trip_loss_portion, 
                                                use_L2_loss)

                                                
    # Actualy train new triplet model for "nb_epochs" epochs
    history_dict, model = trip_utils.train_triplet_model(
                                        trip_model, opt, trip_worker, scratch_model_func,
                                        train_data, test_data, margin, 
                                        expected_loss, batch_size, nb_epochs,
                                        step_size, easy_steps, medium_steps, 
                                        trip_loss_portion, dynamic_margin=dynamic_margin, 
                                        warm_up_examples_per_anchor=warm_up_examples_per_anchor, 
                                        is_L2=use_L2_loss, image_aug=aug_gen, 
                                        callbacks=callbacks)

    # Save history
    if history_filepath:
        pickle.dump(history_dict, open(history_filepath, "wb"))


    # Return history
    return history_dict, model
	

# L1 triplet model training wrapper    
def compile_and_train_L1_trip_model(model, opt, train_data, test_data, 
                        nb_epochs, batch_size, scratch_model_func, trip_worker,
                        margin=trip_l1_margin, expected_loss=expected_loss_val, 
                        trip_loss_percent=trip_loss_percent, 
                        step_size=3, easy_steps=2, medium_steps=2,
                        dynamic_margin=False, warm_up_examples_per_anchor=1,
                        history_filepath=None, aug_gen=None, callbacks=None):
	
    # Simple call to the triplet loss training helper
    return compile_and_train_trip_model_helper(model, opt, train_data, 
                            test_data, nb_epochs, batch_size, scratch_model_func, 
                            trip_worker, False, expected_loss, margin, 
                            trip_loss_percent, step_size, easy_steps, medium_steps,
                            dynamic_margin, warm_up_examples_per_anchor, 
                            history_filepath, aug_gen, callbacks)	
	

# L2 triplet model training wrapper        
def compile_and_train_L2_trip_model(model, opt, train_data, test_data, 
                        nb_epochs, batch_size, scratch_model_func, trip_worker,
                        margin=trip_l2_margin, expected_loss=expected_loss_val, 
                        trip_loss_percent=trip_loss_percent, 
                        step_size=3, easy_steps=2, medium_steps=2,
                        dynamic_margin=False, warm_up_examples_per_anchor=1,
                        history_filepath=None, aug_gen=None, callbacks=None):
    
    # Simple call to the triplet loss training helper
    return compile_and_train_trip_model_helper(model, opt, train_data, 
                            test_data, nb_epochs, batch_size, scratch_model_func, 
                            trip_worker, True, expected_loss, margin, 
                            trip_loss_percent, step_size, easy_steps, medium_steps,
                            dynamic_margin, warm_up_examples_per_anchor, 
                            history_filepath, aug_gen, callbacks)		

                            
                            
####### Main/overarching Snapshot training function call

#   For L1 and L2 triplet loss training, the default parameter values (above) are used
#   However parameters to the triplet loss function can be changed using the "kwargs" dict
#   Specifically, triplet training requires "kwargs" to have a "multi_thread_trip_gen" object 
#       labelled as "trip_worker" 
def conduct_snapshot_training(sequences, train_data, test_data, batch_size,
                                scratch_model_func, weight_dirname, 
                                history_dirname, aug_gen=None, **kwargs):
        
    for cur_seq, init_lrs, iter_epochs, restart_indices in sequences:
        training_started = False
                
        # Add 0 to restart indices to ensure it is there
        if not isinstance(restart_indices, (list, tuple)):
            restart_indices = [0]
        elif not 0 in restart_indices:
            restart_indices.append(0)
            
        # Get base string from sequence of optimizers/loss functions
        base_string = get_checkpoint_base_string(
                                    cur_seq, init_lrs, iter_epochs, restart_indices)
        
        # Iterate through each hyperparameter change
        for i, hyper_parm in enumerate(cur_seq):						
            # Check for previously trained overlapping Snapshots models	
            new_weights_file_name = weight_dirname + base_string + (".%d.hdf5" % i)
            new_history_file_name = history_dirname + base_string + (".%d.pickle" % i)

            old_weights_file_name = training_already_done(cur_seq[:i+1], init_lrs[:i+1], 
                                                            iter_epochs[:i+1], restart_indices[:i+1], 
                                                            i, weight_dirname)

            if old_weights_file_name:
                # Copy other weight file to new location if segment of training already done
                old_history_file_name = (old_weights_file_name[:-5] + 
                                            old_weights_file_name[-5:].replace('hdf5', 'pickle'))

                if ((weight_dirname + old_weights_file_name) != new_weights_file_name):
                    copyfile(weight_dirname + old_weights_file_name, new_weights_file_name)
                    copyfile(history_dirname + old_history_file_name, new_history_file_name)

                # Go to the next Snapshot to potentially begin training
                continue
                
            # Ensure that the weights will be saved
            callbacks = [ ModelCheckpoint(new_weights_file_name,
                            monitor="acc", period=iter_epochs[i],
                            save_best_only=False, save_weights_only=False) ]
            
            # Set Snapshot-specific hyperparameters
            if hyper_parm == ADAM_OPT:
                optimizer = Adam(lr=0.00001)
            
            elif hyper_parm == NADAM_OPT:
                optimizer = Nadam(lr=0.0001)
            
            elif hyper_parm == ADADELTA_OPT:
                optimizer = Adadelta()
            
            elif (hyper_parm == COSINE_SGD_OPT or hyper_parm == TRIP_L1_LOSS or 
                    hyper_parm == TRIP_L2_LOSS):
                    
                optimizer = SGD(lr=init_lrs[i])
                callbacks.append(cb_utils.CosineLRScheduler(init_lrs[i], 
                                                                iter_epochs[i]))
                                    
            else:
                assert False, (("%s:  Not a valid hyperparameter "  % hyper_parm) + \
                                "modification for a Snapshot model")
            
            # Set up the model
            if not training_started or i in restart_indices:
                training_started = True
                model = scratch_model_func()
                
                if not i in restart_indices:
                    prev_weights_file_name = (
                            weight_dirname + base_string + (".%d.hdf5" % (i - 1)))
                            
                    model.load_weights(prev_weights_file_name)

                    
            # Set training function
            if hyper_parm == TRIP_L1_LOSS:
                training_func = compile_and_train_L1_trip_model
            elif hyper_parm == TRIP_L2_LOSS:
                training_func = compile_and_train_L2_trip_model
            else:
                training_func = compile_and_train_single_model
            
            
            # Begin actual compilation and training of the model
            kwargs["scratch_model_func"] = scratch_model_func
            
            _, model = training_func(model, optimizer, train_data, test_data, 
                                        iter_epochs[i], batch_size,
                                        history_filepath=new_history_file_name,
                                        aug_gen=aug_gen, callbacks=callbacks,
                                        **kwargs)

                                        
    # Return nothing
    return