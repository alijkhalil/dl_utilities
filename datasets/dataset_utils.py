# Import statements
import operator

import numpy as np
from keras import backend as K



####### Image Pre-processing Functions

# Simply divide by half of range (e.g. 255) and substract by 1
def simple_image_preprocess(training_data, test_data):
    # Convert to 32-bit floats
    training_data = training_data.astype('float32')
    test_data = test_data.astype('float32')

    # Perform transformation
    training_data /= 127.5
    training_data -= 1
    
    test_data /= 127.5
    test_data -= 1
    
    # Return 0 centered data (ranging from -1 to 1)
    return training_data, test_data
    

# A bit more sophisticated pre-processing based on per-channel mean
def normal_image_preprocess(training_data, test_data):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    # Convert to 32-bit floats
    training_data = training_data.astype('float32')
    test_data = test_data.astype('float32')

    # Perform transformation
    for i in range(training_data.shape[channel_axis]):
        if channel_axis == 1:
            mean_val = np.mean(training_data[:, i])
        else:
            mean_val = np.mean(training_data[:, :, :, i])
        
        training_data[:, :, :, i] -= mean_val
        training_data[:, :, :, i] /= 127.5
        
        test_data[:, :, :, i] -= mean_val
        test_data[:, :, :, i] /= 127.5
    
    # Return 0 centered data (ranging from -1 to 1)
    return training_data, test_data
    


####### Text Processing Functions (mostly for IMDB dataset with RNNs)    

# Convert a sequence of word ID's into a string
def convert_word_list_to_str(word_list, sorted_keys, start_index=4):
    out_string=""
    
    # Iterate through word ID's
    for i, num in enumerate(word_list):
        if num == 1:
            out_string += '^'
        elif num == 2:
            out_string += '*'
        else:
            out_string += sorted_keys[num-start_index][0]
            if i + 1 != len(word_list):
                out_string += ' '
        
    # Add end token
    out_string += '$'
    
    
    # Return it
    return out_string


# Convert a specially formmated string (from 'convert_review_to_str') into an list of chars 
def convert_str_to_char_list(input_str):
    char_ord = ord('a')
    digit_ord = ord('0')

    # Transform string into list of chars (omitting unusual chars)
    char_list = []
    for char in input_str:
        if char == '^':
            char_list.append(1)
        elif char == '$':
            char_list.append(2)
        elif char == ' ':
            char_list.append(3)
        elif char == '\'':
            char_list.append(4)                
        elif char >= 'a' and char <= 'z':
            char_list.append(ord(char) - char_ord + 5)
        elif char >= '0' and char <= '9':
            char_list.append(ord(char) - digit_ord + 31)
    
    
    # Return char review list
    return np.array(char_list)

    