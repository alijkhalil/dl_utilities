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
