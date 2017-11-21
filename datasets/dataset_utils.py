# Import statements
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

special_chars = [ '^', '$', ' ', '\'' ]
alpha_chars = [ chr(ord('a') + i) for i in range(26) ]
digit_chars = [ chr(ord('0') + i) for i in range(10) ]


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
    min_char_ord = ord('a')
    min_digit_ord = ord('0')
    
    special_start = 1
    alpha_start = len(special_chars) + special_start
    digit_start = len(alpha_chars) + alpha_start
    
    def find_index(item, list):
        for i, tmp_item in enumerate(list):
            if item == tmp_item:
                return i
        
        return 0

    # Transform string into list of chars (omitting unusual chars)
    char_list = []
    for char in input_str:
        if char in special_chars:
            char_list.append(find_index(char, special_chars) + special_start)
            
        elif char >= 'a' and char <= 'z':
            char_list.append(ord(char) - min_char_ord + alpha_start)
            
        elif char in digit_chars:
            char_list.append(ord(char) - min_digit_ord + digit_start)
    
    
    # Return char review list
    return char_list
        

# Get possible chars in conversion (with addition of 1 for padding char)
def get_total_possible_chars():
    return (len(special_chars) + 
                len(alpha_chars) + 
                len(digit_chars) + 
                1)
                
    
# Use functions above to convert word-based dataset to char-based one
def convert_word_dataset_to_char_dataset(X_dataset, sorted_keys):
    # Convert to strings
    str_reviews = []
    for i in range(len(X_dataset)):
        str_reviews.append(convert_word_list_to_str(X_dataset[i], sorted_keys))
        
    # Iterate through each review and convert to char-based lists
    char_reviews = []
    for str_review in str_reviews:
        char_reviews.append(convert_str_to_char_list(str_review))
        
    
    # Return text in char sequence format
    return char_reviews

    