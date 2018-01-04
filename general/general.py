# Import statements
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


PADDING_TOKEN_ENCODING = 0
END_TOKEN_ENCODING = 1



# Assumes stride of 1 and any non-valid padding scheme
def get_effective_receptive_field(filter_sizes, dilation_rates):
    prev_effective_rf = 1
    effective_rf = prev_effective_rf
    
    for filter_size, dr in zip(filter_sizes, dilation_rates):
        cur_rf = filter_size * dr - dr + filter_size
        
        effective_rf = prev_effective_rf + cur_rf - 1
        prev_effective_rf = effective_rf
        
    return int(effective_rf)
    
    
# Transform N=rank np_array in one-hot vector
def transform_into_one_hot(init_data, num_poss_values):    
    flat_init_data = init_data.flatten()
    
    # Create new flat np array with all zero's
    orig_shape_flat = flat_init_data.shape[0]    
    new_shape_flat = orig_shape_flat * num_poss_values
    
    ret_arr = np.zeros(new_shape_flat)

    # Mark appropiate indicess with 1 to make them one-hot encoded
    for i in range(orig_shape_flat):
        ret_arr[(i * num_poss_values) + flat_init_data[i]] = 1
        
    
    # Return final one-hot vector (by reshaping it to correct size)
    new_shape = init_data.shape + (num_poss_values, )    
    return np.reshape(ret_arr, new_shape)
    
    
# Remove padding items from a one hot vector so they aren't factored into loss
def remove_padding_from_one_hot(one_hot_input, padding_ID=0):
    # Assumes that padding is 0
    tmp_mask = np.ones_like(one_hot_input[0])
    tmp_mask[:, padding_ID] = 0               
    
    # Mask everything except padding tokens so that they are all 0's
    for i, one_hot in enumerate(one_hot_input):
        one_hot_input[i] = tmp_mask * one_hot

    
    # Return resulting vector
    return one_hot_input

    
# Amplify one-hot encoding of end-token to incresae its loss (relative to other tokens)
#       Should encourage model to more aggresively predict end token (to avoid super long sequences)
def amplify_end_tok_from_one_hot(one_hot_input, amplify_factor, end_token_ID=1):
    # Assumes that end token is 1
    tmp_mask = np.ones_like(one_hot_input[0])
    tmp_mask[:, end_token_ID] = amplify_factor               
    
    # Mask everything except padding tokens so that they are all 0's
    for i, one_hot in enumerate(one_hot_input):
        one_hot_input[i] = tmp_mask * one_hot

    
    # Return resulting vector
    return one_hot_input       
    
    
# Decode encoded sequence and return it
def decode_encoded_seq(enc_seq, decoder):
    char_list = []
    for enc_key in enc_seq:
        char_list.append(decoder[int(enc_key)])
    
    return char_list
    
    
# Conduct beam search based on set on likely results
#       Prev solutions: list of tuples -- (np_array of items in sequential/encoder format, prob)
#       Cur prob: list of probabilities for next item addition to each list of items
#       Cur_index: Index to replace in the solutions
#       Min_end_index: Index to replace in the solutions
#       Beam size: Number of list of items to track
def conduct_beam_search(prev_solutions, cur_probs, 
                            cur_index, min_end_index=None,
                            beam_size=10):

    # Ensure that list of solutions/probabilities are the correct size
    if len(prev_solutions) != beam_size:
        raise ValueError("The 'prev_solutions' variable should have exactly %d "
                            "elements." % beam_size)

    if len(cur_probs) != beam_size:
        raise ValueError("The 'cur_probs' solutions variable should have exactly %d "
                            "elements." % beam_size)  

                      
    if beam_size < 2:
        raise ValueError("The 'beam_size' variable should be at least 2.")  
                      

    # Eliminate duplicate previous solutions (on first iteration of beam search)
    del_indices = []
    for i in range(beam_size):
        for j in range((i+1), beam_size):
            if np.array_equal(prev_solutions[i][0], prev_solutions[j][0]):
                del_indices.append(j)
    
    np_del_indices = np.sort(np.unique(np.array(del_indices)))
    for i, del_index in enumerate(np_del_indices):
            del prev_solutions[del_index - i]
            del cur_probs[del_index - i]
            

    # Set up min sequence length
    if min_end_index is None:
        max_seq_len = prev_solutions[0][0].shape[0]
        min_end_index = int(max_seq_len * 0.5)
        
    # Set up updated solutions to forcably include an option with an end token
    if cur_index < min_end_index:
        max_end_prob = 0.0
    else:
        max_end_prob = prev_solutions[0][1]

    
    # Get best "beam-size" new options (along with their associated probs)
    updated_solutions = [ ]

    update_done = False
    min_prob_i = 0
        
    for prev_sol, new_probs in zip(prev_solutions, cur_probs):
        # Update probabilities to include newest element
        prev_items, prev_prob = prev_sol
        tmp_probs = prev_prob * new_probs
        
        
        # Skip sequence if it has ended already
        start_i = 0
        
        if prev_items[cur_index - 1] == END_TOKEN_ENCODING:
            if len(updated_solutions) < beam_size:
                updated_solutions.append((prev_items, prev_prob))
                
            elif updated_solutions[min_prob_i][1] < prev_prob:
                updated_solutions[min_prob_i] = (prev_items, prev_prob)                
            
            continue
            
            
        # Set up probs to exclude an end and padding tokens        
        start_i = 2
        tmp_probs = tmp_probs[2:]
        
        
        # Ensure that first solution has an end token
        cur_end_prob = tmp_probs[END_TOKEN_ENCODING]
        if len(updated_solutions) == 0:
            prev_items[cur_index] = END_TOKEN_ENCODING
            updated_solutions.append((np.array(prev_items), max_end_prob))        
        
        
        # Check to see if optimal solution with end token can be improved
        if cur_index >= min_end_index and max_end_prob < cur_end_prob:
            prev_items[cur_index] = END_TOKEN_ENCODING
            updated_solutions[0] = (np.array(prev_items), cur_end_prob)

            max_end_prob = cur_end_prob
        
            
        # Change updated solutions list if new solution has a top "beam_search" prob 
        for i, new_p in enumerate(tmp_probs):
            if len(updated_solutions) < beam_size:
                prev_items[cur_index] = i + start_i
                updated_solutions.append((np.array(prev_items), new_p))
                update_done = True 

            else:
                _, min_p = updated_solutions[min_prob_i]
                if min_p < new_p:
                    prev_items[cur_index] = i + start_i
                    updated_solutions[min_prob_i] = ((np.array(prev_items), new_p))
                    update_done = True
                    
            # If update done, get min prob index
            if update_done:
                update_done = False
                
                min_prob_i = 1
                for j in range(1, len(updated_solutions)):
                    _, tmp_p = updated_solutions[j]
                    _, min_p = updated_solutions[min_prob_i]
                    
                    if tmp_p < min_p:
                        min_prob_i = j

    
    # Normalize updated solution probabilities so they never get too small    
    if len(updated_solutions) < beam_size:
        raise ValueError("SOMETHING WRONG %d" % len(updated_solutions))

    prob_sum = sum(top_p for _, top_p in updated_solutions)        
    for i in range(beam_size):
        sol_arr, sol_prob = updated_solutions[i]
        updated_solutions[i] = (sol_arr, float(sol_prob / prob_sum))
        
        
    # Return final list of solutions (with their normalized probabilities
    return updated_solutions
    
    
# Shuffle list of N-rank and seperate into a numpy training and test sets
#   Assumes that all dimensions in input have an equal size for all samples
def split_list_data(input_list, rank=1, 
                        percent_split=0.7, 
                        desired_permutation=None):
                        
    # Get final shape and flatten input
    nd_shape = (len(input_list), )
    flat_shape = len(input_list)
    
    flat_list = [ el for el in input_list ]

    tmp_input = input_list
    for i in range(1, rank):        
        tmp_input = tmp_input[0]

        if (i + 1) != rank and type(tmp_input) is not list:
            raise ValueError("All non-final dimensions must be lists.")
        
        nd_shape += (len(tmp_input), )
        flat_shape *= len(tmp_input)
        
        flat_list = [ el for el in flat_list ]
    
    
    # Copy flattened input and reshape it to correct size
    ret_arr = np.array(flat_list)
    ret_arr = np.reshape(ret_arr, nd_shape)
    
    
    # Shuffle examples
    num_examples = nd_shape[0]
    
    if desired_permutation is not None:
        example_shuffle_order = desired_permutation
    else:
        example_shuffle_order = np.random.permutation(num_examples)
        
    ret_arr = ret_arr[example_shuffle_order]
    
    
    # Return data split into groups
    split_index = int(percent_split * num_examples)
    return ret_arr[:split_index], ret_arr[split_index:]