# Import statements
import itertools, os, math
import numpy as np
import cPickle as pickle

from dl_utilities.snapshots import snapshot_train_utils as snap_train

from keras import metrics
from keras import backend as K

from keras.optimizers import *
from keras.models import Model
from keras.layers.merge import average



##########################  Snapshot evaluation functions ######################	

# Takes a list of snapshot weight files and returns "cur_size" most recent ones
def trim_snap_weight_list(snapshot_files, base_string, cur_size):
	# Ensure that list is at least long enough for trimming
	if len(snapshot_files) < cur_size:
		print("Could not find enough snapshot weight files to " + 
				"build the '%s' ensemble." % base_string)
		return None
	
	# Get all checkpoint files (in ascending order)
	check_pt_nums = []
	for i, filename in enumerate(snapshot_files):
		cur_num = [int(s) for s in filename.split('.') if s.isdigit()][0]
		check_pt_nums.append((filename, cur_num))
	
	check_pt_nums.sort(key=lambda x: x[1])
	snapshot_files = [ filename for filename, _ in reversed(check_pt_nums) ]
	
	
	# Return only newest "cur_size" Snapshots
	return snapshot_files[:cur_size]


# Takes weight files for a set of Snapshot ensembles and saves their evaluation metrics      
def save_basic_ensemble_evals(sequences, test_data, scratch_model_func, 
                                    weight_dirname, eval_dirname, force=False):
                                    
	# Get test data
	x_test, y_test = test_data
	nb_classes = y_test.shape[1]

	# Iterate through all provided Snapshot models
	for cur_seq, init_lrs, iter_epochs, restart_indices in sequences:
		base_string = snap_train.get_checkpoint_base_string(cur_seq, init_lrs, 
													iter_epochs, restart_indices)
		
		max_size = len(cur_seq)
		if max_size < 2:
			continue
		
		# Get biggest sized ensemble and ensemble omitting first model
		for cur_size in range(max_size - 1, max_size + 1):
			eval_filename = eval_dirname + base_string + (".%d.eval" % cur_size)
			
			if not os.path.isfile(eval_filename) or force:
				# Get checkpoint weights
				tmp_chkpt_names = []
				first_checkpt = base_string + ".0.hdf5" 
				
				for filename in os.listdir(weight_dirname):
					if filename.startswith(base_string):
						tmp_chkpt_names.append(weight_dirname + filename)
				
				# Check to ensure that correct number of files were found
				chkpt_names = trim_snap_weight_list(tmp_chkpt_names, 
												base_string, cur_size)
				if not chkpt_names:
					continue
				
				# Get individual models and their averages
				print("\n\nGetting averages of individual Snapshot models... ")
				running_vals = [ 0.0, 0.0, 0.0 ]
				
				chkpt_models = []
				for i, name in enumerate(chkpt_names):
					model = scratch_model_func()
					model.load_weights(name)
					
					for j, cur_layer in enumerate(model.layers):
						cur_layer.name = ("%s_%d_%d" % (base_string, i, j))
					
					# Keep track of model averages
					model.compile(optimizer='adam', loss='categorical_crossentropy', 
								metrics=['accuracy', metrics.top_k_categorical_accuracy])
						
					for j, val in enumerate(model.evaluate(x_test, y_test, batch_size=64, verbose=0)):
						running_vals[j] += (float(val) / cur_size)
					
					# Add checkpoint model to list of models
					chkpt_models.append(model)
				
				# Create Snapshot Ensemble model
				chkpt_inputs = [ model.input for model in chkpt_models ]
				chkpt_outputs = [ model.output for model in chkpt_models ]
				
				# Usage of "average" vs. "keras.layers.merge.Average": 
				#		https://github.com/fchollet/keras/issues/3921
				final_output = average(chkpt_outputs)
				
				snapshot_model = Model(inputs=chkpt_inputs, outputs=final_output)
				snapshot_model.compile(optimizer='adam', loss='categorical_crossentropy', 
								metrics=['accuracy', metrics.top_k_categorical_accuracy])
				
				# Save eval metrics (in eval_dirname)
				print("\n\nEvaluating the following Snapshot Ensemble:  %s" % eval_filename)
				eval_metrics = snapshot_model.evaluate(
						[x_test] * cur_size, y_test, batch_size=12, verbose=1)
				
				print(snapshot_model.metrics_names)
				print(eval_metrics)
				
				running_vals.extend(eval_metrics)
				pickle.dump(running_vals, open(eval_filename, "wb"))
	
	
	# Return nothing
	return
	

# Like "save_basic_ensemble_evals", except with more detailed info about individual Snapshot models in the ensemble
def save_advanced_ensemble_evals(sequences, test_data, scratch_model_func,
                                    weight_dirname, eval_dirname, force=False):
								
	# Get test data
	x_test, y_test = test_data
	nb_classes = y_test.shape[1]	
		
	for cur_seq, init_lrs, iter_epochs, restart_indices in sequences:
		# Check if any of necessary files aren't yet there
		nb_snapshots = len(cur_seq)
		base_string = snap_train.get_checkpoint_base_string(cur_seq, init_lrs, 
													iter_epochs, restart_indices)

		L2_dist_filename = eval_dirname + base_string + (".%d.l2_distances" % nb_snapshots)
		interpool_filename = eval_dirname + base_string + (".%d.interpolation" % nb_snapshots)
		disagree_filename = eval_dirname + base_string + (".%d.disagree" % nb_snapshots)
		
		if (os.path.isfile(L2_dist_filename) and
				os.path.isfile(interpool_filename) and
				os.path.isfile(disagree_filename) and
				not force):
				
			continue
		
		################ Begin evaluation set-up ################
		print("Evaluating '%s' Snapshot model (with %d snapshots)" 
					% (base_string, nb_snapshots))
					
		# Get model checkpoints
		tmp_chkpt_names = []
		for filename in os.listdir(weight_dirname):
			if filename.startswith(base_string):
				tmp_chkpt_names.append(weight_dirname + filename)
		
		# Check to ensure that correct number of files were found
		chkpt_names = trim_snap_weight_list(tmp_chkpt_names, 
											base_string, nb_snapshots)
		if not chkpt_names:
			continue
		
		# Create Snapshot Ensemble model
		chkpt_models = []
		for i, name in enumerate(reversed(chkpt_names)):
			model = scratch_model_func()
			model.load_weights(name)
			
			model.layers[-2].name = "final_embeddings"
			model.layers[-1].name = "predictions"

			chkpt_models.append(model)
		

		################ Get L2 distance between embeddings ################		
		if not os.path.isfile(L2_dist_filename) or force:
			print("Getting L2 distances between final embeddings of each Snapshot model...")
		
			def average_L2_distance(np_arr):
				return np.average(np.sum(np.square(np_arr), axis=-1, keepdims=True))
			
			def get_model_embeddings(model, inputs):
				embed_layer = model.get_layer("final_embeddings")
				embed_dimension = embed_layer.output_shape[-1]				
				
				f = K.function([model.input] + [K.learning_phase()], [embed_layer.output])				

				num_samples = inputs.shape[0]
				final_embeddings = np.ndarray(shape=(num_samples, embed_dimension))
				for i in range(0, num_samples, 32):
					final_embeddings[i:i+32] = f([inputs[i:i+32], 0.])[0]
					final_embeddings[i:i+32] /= np.sqrt(np.sum(
														np.square(final_embeddings[i:i+32]), 
																	axis=-1, keepdims=True))
					
				return final_embeddings
				
			
			# Get each model's embeddings
			embeddings = [ get_model_embeddings(model, x_test) for model in chkpt_models ]

			# Get actual distances between each model			
			dist = {}					
			for i in range(len(embeddings)):
				for j in range((i + 1), len(embeddings)):
					idetifier_string = ("%s_%s" % (i, j))
					dist[idetifier_string] = average_L2_distance(embeddings[i] - embeddings[j])
					print(idetifier_string, dist[idetifier_string])
					
			# Dump the dictionary with L2 embedding distances of each model 
			pickle.dump(dist, open(L2_dist_filename, "wb"))
		
		
		################ Get weight mixture ################		
		if not os.path.isfile(interpool_filename) or force:
			print("Getting interpolation models between each subset of two Snapshot models...")
		
			def is_legit_layer(layer):
				if "Conv2D" in layer.__class__.__name__:
					return True
				elif "Dense" in layer.__class__.__name__:
					return True
				elif "BatchNormalization" in layer.__class__.__name__:
					return True
				else:
					return False

					
			# Create scratch model						
			play_model = scratch_model_func()
			play_model.load_weights(chkpt_names[0])

			# Evaluation interpool of weights
			interpool = {}		
			for i in range(len(chkpt_models)):
				for j in range((i + 1), len(chkpt_models)):
					for mesh_portion in np.arange(0, 1.1, 0.2):
						mesh_portion2 = float(1 - mesh_portion)
						idetifier_string = ("%s_%s_%.2f" % (i, j, mesh_portion))
						
						# Set each layer's weights in the scratch model
						for layer_num in range(len(play_model.layers)):
							layer1 = chkpt_models[i].layers[layer_num]
							layer2 = chkpt_models[j].layers[layer_num]
							
							if is_legit_layer(layer1):
								weights1 = layer1.get_weights()
								weights2 = layer2.get_weights()

								new_weights = []
								for m in range(len(weights1)):
									new_weights.append((mesh_portion * weights1[m]) + 
														(mesh_portion2 * weights2[m]))
															
								# Set as an interpolation of weights from two seperate Snapshots								
								play_model.layers[layer_num].set_weights(new_weights)
								
						# Get evaluation metrics for specific interpolation
						play_model.compile(optimizer='adam', 
									loss='categorical_crossentropy', 
									metrics =['accuracy', metrics.top_k_categorical_accuracy])
						
						eval_metrics = play_model.evaluate(x_test, y_test, batch_size=64, verbose=0)
						interpool[idetifier_string] = eval_metrics 
						print(idetifier_string, interpool[idetifier_string])
						
			# Dump dictionary of intermodel model evaluation metrics list
			pickle.dump(interpool, open(interpool_filename, "wb"))

		
		################ Get disagreement percentage ################		
		if not os.path.isfile(disagree_filename) or force:
			print("Getting disagreement percentage between each subset of two Snapshot models...")			
		
			def top_1_disagree_percent(pred1, pred2):
				pred1_index = np.argmax(pred1, axis=-1)
				pred2_index = np.argmax(pred2, axis=-1)
				
				total_examples = pred1.shape[0]
				total_same = np.sum(np.equal(pred1_index, pred2_index))
				total_diff = total_examples - total_same
			
				return (float(total_diff) / total_examples)
			
			def top_k_disagree_percent(pred1, pred2, k=5):
				top_index = (k * -1)
				total_examples = pred1.shape[0]
				
				sum = 0
				for i in xrange(total_examples):
					# Get top K indices for each set of predictions
					top_k1 = pred1[i].argsort()[top_index:][::-1]
					top_k2 = pred2[i].argsort()[top_index:][::-1]	
					
					# Check if any of the top K are different
					is_different = False
					for j in range(k):
						found = False
						for m in range(k):
							if top_k1[j] == top_k2[m]:
								found = True
								break
						
						if not found:
							is_different = True
							break
							
					sum += is_different
					
				return (float(sum) / total_examples)
			
			def get_predictions(model, test_images):
				pred_layer = model.layers[-1]				
				f = K.function([model.input] + [K.learning_phase()], [pred_layer.output])	
			
				num_examples = test_images.shape[0]
				model_predictions = np.ndarray(shape=(num_examples, nb_classes))
				
				# Perform inference for predictions
				for i in range(0, num_examples, 128):		
					model_predictions[i:i+128] = f([test_images[i:i+128], 0.])[0]
				
				return model_predictions				
			
			
			# Get each model's prediction as a numpy array
			predictions = [ get_predictions(model, x_test) for model in chkpt_models ]

			# Loop through each predict to store disagreement percent 
			disagree = {}		
			for i in range(len(predictions)):
				for j in range((i + 1), len(predictions)):
					idetifier_string = ("%s_%s" % (i, j))
					disagree[idetifier_string] = (top_1_disagree_percent(predictions[i], predictions[j]),
													top_k_disagree_percent(predictions[i], predictions[j], k=3))
					print(idetifier_string, disagree[idetifier_string])

			# Dump the distance dictionary of tuples
			pickle.dump(disagree, open(disagree_filename, "wb"))
			print("\n")
	
	# Return nothing
	return None
	


	
##########################  Snapshot Debugging Functions ######################


# Prints already calculated evaluation metrics (from saved files in "eval_dirname")
def print_all_eval_info(sequences, scratch_model_func, 
                            eval_dirname, advanced_metrics=False):
							
	# Create a temporary model just to print metrics
	tmp_model = scratch_model_func()
	tmp_model.compile(optimizer='adam', loss='categorical_crossentropy', 
							metrics=['accuracy', metrics.top_k_categorical_accuracy])
	
	# Print for each sequence with each number of Snapshots
	started = False
	for cur_seq, init_lrs, iter_epochs, restart_indices in sequences:
		max_size = len(cur_seq)
		if max_size < 2:
			continue
		
		# Get biggest sized ensemble and ensemble omitting first model
		for cur_size in range(max_size - 1, max_size + 1):
			# Print divider between points
			if not started:
				started = True
			else:
				print("\n==========\n")
			
			# Get base string for the sequence
			base_string = snap_train.get_checkpoint_base_string(cur_seq, init_lrs, 
														iter_epochs, restart_indices)
														
			print("\nDisplaying evaluation information for '%s' (%d snapshots)\n\n" %
					(base_string, cur_size))
		
			# Start with the basic evalution
			eval_filename = eval_dirname + base_string + (".%d.eval" % cur_size)
			if os.path.isfile(eval_filename):
				metrics_len = len(tmp_model.metrics_names)
				eval_metrics = pickle.load(open(eval_filename, "rb"))				
				
				print("Averages of underlying Snapshot models validation metrics:")
				print(tmp_model.metrics_names)
				print(eval_metrics[:metrics_len])	
				
				print("Snapshot Ensemble validation metrics:")
				print(tmp_model.metrics_names)
				print(eval_metrics[metrics_len:])	

				print("\n")
			
			if advanced_metrics:
				# Next, print L2 distances
				eval_filename = eval_dirname + base_string + (".%d.l2_distances" % cur_size)
				if os.path.isfile(eval_filename):
					eval_metrics = pickle.load(open(eval_filename, "rb"))
					
					print("Index_1, Index_2:  distance_between_embeddings")
					for cur_key in sorted(eval_metrics.iterkeys()):
						print("%s:" % cur_key, eval_metrics[cur_key])
					print("\n")

				# Next, print interpolation metrics
				eval_filename = eval_dirname + base_string + (".%d.interpolation" % cur_size)
				if os.path.isfile(eval_filename):
					eval_metrics = pickle.load(open(eval_filename, "rb"))
					
					print("Index_1, Index_2, percent_index_1:  " + ', '.join(tmp_model.metrics_names))
					for cur_key in sorted(eval_metrics.iterkeys()):
						print("%s:" % cur_key, eval_metrics[cur_key])				
					print("\n")

				# Next, print disagreement metrics	
				eval_filename = eval_dirname + base_string + (".%d.disagree" % cur_size)
				if os.path.isfile(eval_filename):
					eval_metrics = pickle.load(open(eval_filename, "rb"))
					
					print("Index_1, Index_2:  (top_1_disagreement, top_3_disagreement)")
					for cur_key in sorted(eval_metrics.iterkeys()):
						print("%s:" % cur_key, eval_metrics[cur_key])				
					print("\n")

	# Return nothing
	return None
	

# Shows training history for Snapshot models in an ensemble	
def print_snapshot_model_hist(sequences, history_dirname, last_only=False):
	for cur_seq, init_lrs, iter_epochs, restart_indices in sequences:
		# Get mathcing history pickle files
		base_string = snap_train.get_checkpoint_base_string(cur_seq, init_lrs, 
											iter_epochs, restart_indices)
		
		tmp_matching_files = []
		for filename in os.listdir(history_dirname):
			if filename.startswith(base_string):
				tmp_matching_files.append(history_dirname + filename)
		
		# Sort in reverse order
		num_snaps = len(tmp_matching_files)
		matching_files = trim_snap_weight_list(tmp_matching_files, base_string, 
													num_snaps)
		
		# Print history of each one
		print("Showing history for '%s'\n" % base_string)
		for hist_file in reversed(matching_files):
			hist = pickle.load(open(hist_file, "rb"))
			index_val = [int(s) for s in hist_file.split('.') if s.isdigit()][0]
			
			print("History for index:  %d" % index_val)
			for key in hist.keys():
				print("%s: " % key)
				if last_only:
					print(hist[key][-1])
				else:
					print(hist[key][::5])
			print("\n")
			
		print ("\n======\n\n")
	
	
	# Return nothing
	return None
	

# Display evaluation metrics for an ensemble of arbitary models (without saving anything)	
def show_basic_eval_for_snapshot_combo(snapshot_weight_list, scratch_model_func, 
                                        weight_dirname, test_data):
                                        
	# Seperate test_data
	x_test, y_test = test_data
	nb_classes = y_test.shape[1]

	# Create Snapshot Ensemble model
	chkpt_models = []
	for i, filename in enumerate(snapshot_weight_list):
		model = scratch_model_func()
		model.load_weights(weight_dirname + filename)
		
		for j, cur_layer in enumerate(model.layers):
			cur_layer.name = ("tmpy_%d_%d" % (i, j))
		
		chkpt_models.append(model)
			
	chkpt_inputs = [ model.input for model in chkpt_models ]
	chkpt_outputs = [ model.output for model in chkpt_models ]
	
	if len(chkpt_models) > 1:
		final_output = average(chkpt_outputs)
	else:
		final_output = chkpt_outputs[0]
		
	snapshot_model = Model(inputs=chkpt_inputs, outputs=final_output)
	snapshot_model.compile(optimizer='adam', loss='categorical_crossentropy', 
                            metrics=['accuracy', metrics.top_k_categorical_accuracy])
	
	# Display eval metrics
	snapshot_size = len(snapshot_weight_list)
	print("\nEvaluating the %d model shotshot" % snapshot_size)
	print(snapshot_weight_list)
	eval_metrics = snapshot_model.evaluate(
			[x_test] * snapshot_size, y_test, 
			batch_size=int(math.ceil(48 / snapshot_size)), 
			verbose=1)
	
	print(snapshot_model.metrics_names)
	print(eval_metrics)
	
	
	# Return nothing
	return None
			
