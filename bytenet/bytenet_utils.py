# Import statements
import os, gc, sys, time
import numpy as np
import math, random

import Queue
import warnings, multiprocessing, threading
from collections import deque
        
import tensorflow as tf
        
from keras import backend as K
from keras import models, metrics

from dl_utilities.callbacks import callback_utils as cb_utils
from dl_utilities.general import general as gen_utils



# Static variables
ORIG_TOK='x'
MODEL_GEN_TOK='o'  

DIFF_SEQS = [[ORIG_TOK], 
                [MODEL_GEN_TOK, ORIG_TOK, ORIG_TOK, ORIG_TOK],
                [MODEL_GEN_TOK, ORIG_TOK, ORIG_TOK],
                [MODEL_GEN_TOK, ORIG_TOK],
                [MODEL_GEN_TOK, MODEL_GEN_TOK, ORIG_TOK],
                [MODEL_GEN_TOK, MODEL_GEN_TOK, MODEL_GEN_TOK, ORIG_TOK],
                [MODEL_GEN_TOK]]
MAX_DIFF_LEVEL = len(DIFF_SEQS) - 1




##################  ByteNet sequence generator  ###################      
        
        
# Internal simple range generator class for ability to change current index
class RangeGen(object): 
    def __init__(self, start_i, end_i, skip_val):		
        self.cur_i = start_i
        self.end_i = end_i
        self.skip_val = skip_val
     
    def reset_cur_index(self, new_i):
        self.cur_i = new_i - self.skip_val
        
    def get_generator(self):
        while self.cur_i < self.end_i:
            yield self.cur_i
            self.cur_i += self.skip_val    
    

    
# Wrapper functions for get_shifted_predictions
#       Notes:
#             -All inputs, targets, and hints should be numpy arrays
#             -Func variables should obviously be functions
#             -Model variable is a BN model with current weights
#             -All other variables are integer lengths and sizes
def generator_pred_wrapper(model, difficulty_level, 
                            batch_size,
                            num_examples,                                                    
                            encoder_inputs,
                            get_encoder_func, 
                            shifted_inputs, 
                            get_shifted_preds_func,
                            final_str_len_hints,    
                            actual_targets,        
                            input_processing_func,                  
                            label_conversion_func,                  
                            context_len, init_beam_size):
    
    # Get only necessary subset
    tmp_permutation = np.random.permutation(shifted_inputs.shape[0])[:num_examples]
        
    shifted_inputs = shifted_inputs[tmp_permutation]
    final_str_len_hints = final_str_len_hints[tmp_permutation]
    actual_targets = actual_targets[tmp_permutation]
    
    
    # If applicable get output of encoder to pass to decoder
    encoder_outputs = None 
    if encoder_inputs is not None:
        enc_model = get_encoder_func(model)
        
        encoder_inputs = encoder_inputs[tmp_permutation]
        encoder_outputs = enc_model.predict(encoder_inputs, batch_size=batch_size) 
    
    
    # The "input_processing_func" needs to take model variable 
    def processing_helper_func(shifted_inputs, encoder_outputs, model=model):
        return input_processing_func(shifted_inputs, encoder_outputs, model)


    # Get predictions
    preds = get_shifted_preds_func(shifted_inputs, 
                                        final_str_len_hints, encoder_outputs,
                                        actual_targets, processing_helper_func, 
                                        context_len, difficulty_level, 
                                        (init_beam_size+difficulty_level))
    
    
    # Convert actual targets into one_hot_labels
    one_hot_tgts = label_conversion_func(actual_targets)

    
    # Returnt them and permutation
    return encoder_inputs, preds, one_hot_tgts
    

    
# Sequence generator for cirriculum learning     
class multi_thread_seq_gen(object):
    def __init__(self, queue_size=512, nthreads=8):		
        # Set-up presistent global/shared variables
        self.activity_signal = multiprocessing.Value('i', 0)
        self.any_inside = multiprocessing.Value('i', 0)
        
        self.manager = multiprocessing.Manager()
        self.weight_list = self.manager.list()
        self.weight_lock = multiprocessing.Lock()
        self.weight_callback = cb_utils.UpdateSharedWeights(self.weight_list, self.weight_lock)
                                            
        self.thread_list = []		
        self.q = multiprocessing.Queue(maxsize=queue_size)			
        
        self.parameter_list = self.manager.dict()
        self.difficulty = multiprocessing.Value('i', 0)
        self.is_one_processing = multiprocessing.Value('i', 0)
        
        
        # Helper function for approximately determining fullness of queue
        def is_queue_full(queue):
            cur_size = int(queue.qsize())
            aprox_full_size = int(queue_size * 0.85)
            
            if cur_size > aprox_full_size:
                return True
            else:
                return False
                
        
        # Main worker thread function
        def worker(work_sig, qlist, wlist, wlock,
                        synced_difficulty, param_list, 
                        any_inside, is_one_working):
            # Set up envirnoment and session
            # Use the "nvidia-smi" command for more info on GPU memory usage
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            
            pid_val = os.getpid()
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

                        
                # Initialize dummy model and function to get final embedding layers
                K.clear_session()				
                                           
                                           
                # If circulum training is ready, set up parameters
                scratch_model_func = param_list['new_model_func']
                tmp_model = scratch_model_func()                
                
                
                # Set up predictions function
                predictions_func = param_list['predictions_func']                

                
                # Get size of subset of examples
                subset_num_examples = param_list['subset_num_examples']
                                
                                
                # Iterate through samples until enough triplets generated for training
                alt_shifted_predictions = None
                do_normal_training = True
                
                cur_batch_index = 0
                with synced_difficulty.get_lock():
                    cur_difficulty = int(synced_difficulty.value) 
                    
                    
                while True:
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
                        
                    # Get current difficulty
                    with synced_difficulty.get_lock():
                        orig_difficulty = int(synced_difficulty.value)
                        
                    # If difficulty changed recently, try to use previously calculated predictions
                    if cur_difficulty != orig_difficulty:
                        check_alt = True
                        
                        if not do_normal_training:
                            with is_one_working.get_lock():
                                is_one_working.value = (is_one_working.value - 1)
                            
                            do_normal_training = True                                
                    else:
                        check_alt = False
                        
                
                    # Increment inside/"still working" counter before getting predictions	
                    with any_inside.get_lock():
                        any_inside.value = (any_inside.value + 1)           
                        
                    
                    # Call predictions function with correct difficulty
                    if do_normal_training:
                        cur_batch_index = 0
                        
                        # Do not recalculate predictions if they are available 
                        if check_alt and alt_shifted_predictions is not None:
                            tmp_encoder_inputs = alt_tmp_encoder_inputs
                            shifted_predictions = alt_shifted_predictions
                            tmp_labels = alt_tmp_lables
                            
                            alt_shifted_predictions = None

                        else:
                            ret_vals = predictions_func(tmp_model, orig_difficulty)
                            tmp_encoder_inputs, shifted_predictions, tmp_labels = ret_vals
                        
                    else:
                        # Use cur difficulty in case difficulty has changed recently
                        ret_vals = predictions_func(tmp_model, (orig_difficulty + 1))
                        alt_tmp_encoder_inputs, alt_shifted_predictions, alt_tmp_lables = ret_vals                        
                        
                        with is_one_working.get_lock():
                            is_one_working.value = (is_one_working.value - 1)
                        
                        do_normal_training = True
                    
                    
                    # Get batch size and initiate generator (mimicking range)
                    batch_size = self.parameter_list['batch_size']
                    
                    range_gen = RangeGen(cur_batch_index, subset_num_examples, batch_size)
                    custom_range_generator = range_gen.get_generator()               
                    
                    
                    # Begin add examples to the queue                        
                    for start_i in custom_range_generator:
                        # Ensure that queue is still being filled with correct difficulty
                        if start_i % (4 * batch_size) == 0:
                            with synced_difficulty.get_lock():
                                cur_difficulty = int(synced_difficulty.value)            

                        if cur_difficulty != orig_difficulty: 
                            if alt_shifted_predictions is not None:
                                tmp_encoder_inputs = alt_tmp_encoder_inputs
                                shifted_predictions = alt_shifted_predictions
                                tmp_labels = alt_tmp_lables
                                
                                alt_shifted_predictions = None

                                range_gen.reset_cur_index(0)
                                orig_difficulty = cur_difficulty
                                                                
                                continue
                            
                            else:
                                break
                                
                                
                        # Put items in the queue  
                        inputs = [ ]
                        outputs = [ tmp_labels[start_i:start_i+batch_size] ]
                        
                        if tmp_encoder_inputs is not None:
                            inputs.append(tmp_encoder_inputs[start_i:start_i+batch_size])
                        
                        inputs.append(shifted_predictions[start_i:start_i+batch_size])                        
                        
                        qlist.put((inputs, outputs))
                        

                        # See if it makes sense to pre-emptively get next difficulty tokens
                        if alt_shifted_predictions is None and is_queue_full(qlist) and False:
                            with is_one_working.get_lock():
                                if is_one_working.value == 0:
                                    is_one_working.value = (is_one_working.value + 1)                            
                                    do_normal_training = False
                                    
                            if not do_normal_training:         
                                cur_batch_index = start_i
                                break
                            
                            
                    # Decrement inside counter
                    with any_inside.get_lock():
                        any_inside.value = (any_inside.value - 1)
                            
                            
        # Start "nthreads" processes to concurrently yield triplets
        for _ in range(nthreads):
            t = multiprocessing.Process(target=worker, args=(
                                                self.activity_signal, self.q, 
                                                self.weight_list, self.weight_lock, 
                                                self.difficulty, self.parameter_list, 
                                                self.any_inside, self.is_one_processing))
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


    # Set new difficulty as model becomes more tailored to triplet loss
    def set_new_difficulty_value(self, new_difficulty):
        with self.difficulty.get_lock():
            self.difficulty.value = new_difficulty


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
    #     Notes:
    #           -The 'scratch_model_func' should:
    #               -takes no parameters 
    #               -return the model used during training
    #           -The 'predictions_func' should:
    #               -takes model and difficulty level
    #               -returns tuple of inputs, predictions, and labels (in that order)
    #            -Subset examples is on per-thread basis
    #               -must be less total_num_examples
    def start_activity(self, scratch_model_func, 
                            predictions_func, 
                            subset_num_examples, 
                            total_num_examples, 
                            batch_size, 
                            difficulty=0):
                
        # Check to ensure that there is no residuals activity from last call to workers	
        if not self.is_ready_for_start():
            return False
        
        
        # Save parameters		        
        self.parameter_list['new_model_func'] = scratch_model_func
        self.parameter_list['predictions_func'] = predictions_func
        
        self.parameter_list['batch_size'] = batch_size
        self.parameter_list['subset_num_examples'] = subset_num_examples

        
        # Ensure that parameters values are acceptable
        if total_num_examples < subset_num_examples:
            raise ValueError("The 'subset_num_examples' variable should not be greater " 
                                    "than the size of 'labels' variable.")
        
        if subset_num_examples % batch_size != 0:
            raise ValueError("The 'subset_num_examples' variable should be divisible by " 
                                    "the 'batch_size' variable.")
            
            
            
        # Set correct difficulty and reset next difficulty processing flag
        with self.difficulty.get_lock():
            self.difficulty.value = difficulty
            
        with self.is_one_processing.get_lock():
            self.is_one_processing.value = 0
            
            
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

        
        
        

        
        
        
##########   Other Model-specific evaluation functions   ###########


# Get length of encoded string
def get_encoded_str_len(encoded_input_seq):
    for i, enc_val in enumerate(encoded_input_seq):
        if enc_val == gen_utils.END_TOKEN_ENCODING:
            return i
        
    return i

    

# Get hot-one encoding of the target char's for evaluation    
def one_hot_conversion(orig_tgt, total_num_char, amp_factor=1):
    # Use general utilities functions to convert numpy array
    one_hot_tgt = gen_utils.transform_into_one_hot(orig_tgt, total_num_char)    
    one_hot_tgt = gen_utils.remove_padding_from_one_hot(one_hot_tgt)
    one_hot_tgt = gen_utils.amplify_end_tok_from_one_hot(one_hot_tgt, amp_factor)

    
    # Return one hot tgt
    return one_hot_tgt
    
    
    
# Gets the components of the ByteNet encoder-decoder framework
def get_subcomponents_of_enc_decode_bytenet(model):
    # Encoding component model
    encode_out_layer = model.get_layer("encode_out")
    encode_model = Model(inputs=model.input[0], outputs=encode_out_layer.output)
    
    # Target embedding (to concatenate with encoder output) model
    target_embed_layer = model.get_layer("target_embed")
    target_embed_model = Model(inputs=model.inputs[1], 
                                outputs=target_embed_layer.output)
    
    # Decoder component model
    decoder_layer = model.get_layer("decoder_model")
    decoder_model = Model(inputs=decoder_layer.inputs[0], 
                            outputs=decoder_layer.outputs[0])


    # Return all three models
    return (encode_model, target_embed_model, decoder_model)
         

         
# Gets char predictions from source language and encoder embedding
def get_prediction_of_enc_decode_bytenet(shifted_targets, encoder_output, 
                                            tgt_model, decoder_model, 
                                            batch_size=None):
                                            
    target_embedding = tgt_model.predict(shifted_targets, batch_size=batch_size)
    
    combined_embedding = np.concatenate((encoder_output, target_embedding), axis=-1)
    prediction = decoder_model.predict(combined_embedding, batch_size=batch_size)

    return prediction                                     
         
         
         
# Function goal:  To provide a generic function to return predictions 
#                           from ByteNet encoder/decoder framework
#
# Shifted_inputs should be:
#       Initial context for character sequence or translation (e.g. first chars in sequence/translation)
#       Should be shifted by right 1 place (so it starts with 0 and ends with one or more 1's)
#
# Final_str_len_hints should be:
#       None or value approximating the desired length of target output strings (as a list)
#
# Encoder_outputs can be either:
#       None or encoder output embeddings for each example (as numpy array)
#
# Actual_targets should be:
#       None or expected output of the "get_predictions" function (as a numpy array)
#
# Input_processing_func should follow these guidelines:
#       Prototype:  processing_func(shifted_inputs, encoder_output=None, model-None)
#       Parameters (from above):  'Shifted_inputs' and optional 'encoder_output'
#       Output:  Character predictions based on inputs
#
# Difficulty_level should be:
#       Less 6, inclusive, in increase order of training difficulty
#       0 - returns actual_targets (without use of processing function) (e.g. teacher forcing) 
#       6 - returns all model-based predictions
#       Negative - same as 6 (e.g. maximum difficulty)
#
def get_shifted_predictions(shifted_inputs, 
                                final_str_len_hints, encoder_outputs,
                                actual_targets, input_processing_func, 
                                start_index, difficulty_level, 
                                beam_size=10):
                                
    # Check difficulty level for correct value
    if difficulty_level < 0:
        difficulty_level = MAX_DIFF_LEVEL
        
    elif difficulty_level == 0:
        copy_arr = np.array(actual_targets)
        ret_arr = np.zeros_like(actual_targets)
        
        ret_arr[:, 1:] = copy_arr[:, :-1]
        return ret_arr
        
    elif difficulty_level > MAX_DIFF_LEVEL:
        raise ValueError("The 'difficulty_level' parameter must be between 0 and %d, "
                            "inclusive."  % MAX_DIFF_LEVEL)
    
    
    # Ensure that actual_targets is not None when it is needed   
    if difficulty_level < MAX_DIFF_LEVEL and actual_targets is None:
        raise ValueError("If the difficulty level is less than %d, you must provide "
                                "the 'actual_targets' array." % MAX_DIFF_LEVEL)
    
        
    # To perform beam search need to start with duplicate of each context
    seq_poss = np.ndarray(shape=((beam_size, ) + shifted_inputs.shape))
    for i in range(beam_size):
        seq_poss[i] = shifted_inputs
        
            
    # Get remainder of the input after context
    test_num_examples = shifted_inputs.shape[0]
    max_time_series_len = shifted_inputs.shape[-1]

    prev_solutions = []
        
    next_chars_arr = None
    diff_seq_arr = DIFF_SEQS[difficulty_level]
    
    for i in range(start_index + 1, max_time_series_len):
        cur_op_i = ((i - start_index - 1) % len(diff_seq_arr))
        
        if diff_seq_arr[cur_op_i] == MODEL_GEN_TOK:
            # Get softmax for next potential chars in each of the "beam_size" possibilities
            for j in range(beam_size):
                cur_seqs = seq_poss[j]
                next_chars = input_processing_func(cur_seqs, encoder_outputs)
                
                if next_chars_arr is None:
                    total_num_char = next_chars.shape[-1]
                    next_chars_arr = np.ndarray(shape=(beam_size, test_num_examples, total_num_char))
                    
                next_chars_arr[j] = next_chars[:, (i - 1), :]

            # Use those softmax probabilities to conduct beam search per each example
            for m in range(test_num_examples): 
                gc.collect()
                
                if len(prev_solutions) < test_num_examples:
                    prev_solutions.append([ (seq, 1.0) for seq in seq_poss[:, m] ])
                
                cur_probs = [ probs for probs in next_chars_arr[:, m, :] ]
                cur_index = i
                
                if final_str_len_hints is not None:
                    tgt_len_low_bound = int(final_str_len_hints[m] - 10)
                else:
                    tgt_len_low_bound = int(max_time_series_len - 30)
                    
                prev_solutions[m] = gen_utils.conduct_beam_search(prev_solutions[m], cur_probs, 
                                                                    cur_index, 
                                                                    min_end_index=tgt_len_low_bound,
                                                                    beam_size=beam_size)
                    
        else:
            # Get probs for each possible next char
            next_char_probs = []
            for j in range(beam_size):
                next_char_probs.append(input_processing_func(seq_poss[j], encoder_outputs))
            
            
            # Simply plug in correct next character if sequences not yet done
            for m in range(test_num_examples):
                next_char = actual_targets[m, i - 1]
                if next_char == gen_utils.PADDING_TOKEN_ENCODING:
                    next_char = gen_utils.END_TOKEN_ENCODING
                
                # For each of the beam size possibilities, update with correct char
                new_sols = []
                for j in range(beam_size):
                    sol_arr, prob = prev_solutions[m][j]
                    
                    if sol_arr[i - 1] != gen_utils.END_TOKEN_ENCODING:
                        sol_arr[i] = next_char
                        prob *= next_char_probs[j][m, (i - 1), next_char]
                    
                    new_sols.append((np.array(sol_arr), prob))
                
                # Normalize new solution probabilities
                prob_sum = sum(top_p for _, top_p in new_sols)        
                for j in range(beam_size):
                    sol_arr, sol_prob = new_sols[j]
                    new_sols[j] = (sol_arr, float(sol_prob / prob_sum))                
                
                # Update solutions
                prev_solutions[m] = new_sols

                
        # Update sequence possibilities with new set of top solutions      
        for m in range(test_num_examples):
            for j in range(beam_size):
                solution_vals, _ = prev_solutions[m][j]
                seq_poss[j, m] = solution_vals
            
            
    # Get optimal solutions
    predicted_solutions = np.ones_like(shifted_inputs)
    for m in range(test_num_examples):
        # Sort in increasing order
        tmp_sorted_solutions = sorted(prev_solutions[m], key=lambda x: x[1])
        
        # Get top encoded solution and set it in the current position
        optimal_enc_solution = tmp_sorted_solutions[-1][0]
        predicted_solutions[m] = optimal_enc_solution
        
        # Pad the end for training purposes
        for i, ch in enumerate(predicted_solutions[m]):
            if ch == 1:
                break
        
        predicted_solutions[m, (i+1):] = 0
        
        
    # Return predicted solutions (already shifted and ready for training use)
    return predicted_solutions


    
# Function to do qualitative evaluation of character generation    
def do_char_gen_evaluation(model, test_tokens, test_num_examples, 
                                context_len, beam_size, 
                                total_num_char, decoder):    
                            
    # Check that num_examples isn't too large                
    if test_num_examples > test_tokens.shape[0]:
        raise ValueError("Requested too many examples to print.")
    
    
    # Shift train/test input over 1 to get labels
    input_seqs = np.ones(shape=((test_num_examples, ) + test_tokens.shape[1:]))
    for i in range(test_num_examples):
        for j in reversed(range(1, context_len + 1)):
            input_seqs[i, j] = test_tokens[i, j - 1]
        input_seqs[i, 0] = 0

    
    # Get prediction array
    shifted_predictions = get_shifted_predictions(input_seqs, None, None, 
                                                    test_tokens, 
                                                    (lambda x, y: model.predict(x)), 
                                                    context_len, -1, beam_size)


    # Decode previous solutions and print first couple of them
    for i, optimal_enc_solution in enumerate(shifted_predictions):
        # Decode the original sequence and prediction too
        ch_list_act_target = gen_utils.decode_encoded_seq(test_tokens[i], decoder)
        ch_list_target = gen_utils.decode_encoded_seq(optimal_enc_solution, decoder)
        
        actual_tgt_string = ""
        final_tgt_string = ""
        
        for act_tgt_ch_int, tgt_ch_int in zip(ch_list_act_target, ch_list_target):
            if act_tgt_ch_int > 1:
                actual_tgt_string += chr(act_tgt_ch_int)
                
            if tgt_ch_int > 1:    
                final_tgt_string += chr(tgt_ch_int)
        
        
        # Print resulting strings (for qualitative assessment)
        print("\n===\n\nExample %d (context in brackets):\n" % i)
        
        print("Actual sequence:  %s" % actual_tgt_string)
        if context_len > 0:
            print("Predicted sequence:  [%s] %s" % (final_tgt_string[:context_len], 
                                                        final_tgt_string[context_len:]))
        else:
            print("Predicted sequence:  %s" % final_tgt_string)

            
            
# Function to do qualitative evaluation of translation 
def do_translation_evaluation(model, src_tokens, tgt_tokens, 
                                test_num_examples, context_len, 
                                beam_size, total_num_char, decoder):    

    # Check that num_examples isn't too large
    if test_num_examples > src_tokens.shape[0]:
        raise ValueError("Requested too many examples to print.")
    
    if tgt_tokens is not None:
        if test_num_examples > tgt_tokens.shape[0]:
            raise ValueError("Requested too many examples to print.")
            
    
    # Shift train/test input over 1 to get labels
    time_steps = src_tokens.shape[-1]
    translated_seqs = np.ones(shape=(test_num_examples, time_steps))

    if tgt_tokens is not None:
        for i in range(test_num_examples):
            for j in reversed(range(1, context_len + 1)):
                translated_seqs[i, j] = tgt_tokens[i, j - 1]
            translated_seqs[i, 0] = 0
    
    else:
        context_len = 0
        for i in range(test_num_examples):
            translated_seqs[i, 0] = 0
            
            
    # Get source string length for a clue on the correct min length of the translation                        
    encode_str_lens = []
    for tok in src_tokens:
        encode_str_lens.append(get_encoded_str_len(tok))

            
    # Seperate model into various components (e.g. encoder/decoder/target embeddings)
    model_components = get_subcomponents_of_enc_decode_bytenet(model)
    encode_model, target_embed_model, decoder_model = model_components
    
    
    # Wrapper for 'get_prediction_of_enc_decode_bytenet' so it's the correct prototype
    def get_pred_wrapper(shifted_targets, encoder_output, 
                               tgt_model=target_embed_model,
                                decoder_model=decoder_model):
                                
        return get_prediction_of_enc_decode_bytenet(shifted_targets, encoder_output,
                                                        tgt_model, decoder_model)                  

        
    # Reusuable encoder outputs        
    encoder_out = encode_model.predict(src_tokens[:test_num_examples])    
        
        
    # Get prediction array
    shifted_predictions = get_shifted_predictions(translated_seqs, encode_str_lens, 
                                                    encoder_out, tgt_tokens, 
                                                    get_pred_wrapper, 
                                                    context_len, -1, beam_size)
            
            
    # Decode previous solutions and print first couple of them
    for i, optimal_enc_solution in enumerate(shifted_predictions):
        # Decode source, true target, and prediction 
        ch_list_src = gen_utils.decode_encoded_seq(src_tokens[i], decoder)
        ch_list_act_target = gen_utils.decode_encoded_seq(tgt_tokens[i], decoder)
        ch_list_target = gen_utils.decode_encoded_seq(optimal_enc_solution, decoder)
        
        final_src_string = ""
        actual_tgt_string = ""
        final_tgt_string = ""
        
        for src_ch_int, act_tgt_ch_int, tgt_ch_int in \
            zip(ch_list_src, ch_list_act_target, ch_list_target):
            
            if src_ch_int > 1:
                final_src_string += chr(src_ch_int)

            if act_tgt_ch_int > 1:
                actual_tgt_string += chr(act_tgt_ch_int)
                
            if tgt_ch_int > 1:    
                final_tgt_string += chr(tgt_ch_int)
        
        
        # Print examples (for qualitative assessment)
        print("\n===\n\nExample %d (context in brackets):\n" % i)
  
        print("Source:  %s  (%d)\n" % (final_src_string, len(final_src_string)))
        print("Actual Translation:  %s  (%d)\n" % (actual_tgt_string, len(actual_tgt_string)))
        
        if context_len > 0:
            print("Machine Translation:  [%s] %s  (%d)" % 
                    (final_tgt_string[:context_len], final_tgt_string[context_len:],
                        len(final_tgt_string)))
        else:
            print("Machine Translation:  %s" % final_tgt_string)
        