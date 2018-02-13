# Import statements
import json
import git
import os
import numpy as np

from PIL import Image

from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical



# Global variables
cur_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
dataset_dir = cur_dir + 'nlvr/'

train_json = dataset_dir + 'train/train.json'
train_img_dir = dataset_dir + 'train/images'

test_json = dataset_dir + 'dev/dev.json'
test_img_dir = dataset_dir + 'dev/images'



# Get GIT repo for Cornell NLVR dataset
def get_cornell_nlvr_repo():
    if not os.path.isdir(dataset_dir):
        print("Getting Cornell NLVR dataset...")
        git.Git(cur_dir).clone("https://github.com/clic-lab/nlvr.git")
        print("Done.")
    
    
# Gets raw_data and tokenizer from JSON file
def get_basic_data(json_file):        
    # Check for existence of JSON file
    if not os.path.exists(json_file):
        raise ValueError("The needed JSON file does not exist in the GIT repo.")

    
    # Get initial dataset list using JSON file
    j_file = open(json_file, 'r')

    data = []
    for record in j_file:
        jn = json.loads(record)
        
        s = str(jn['sentence'])
        idn = jn['identifier']
        la = int(jn['label'] == 'true')
        
        data.append([idn, s, la])
    
    
    # Return final data
    return data
    

# Get a tokenizer based on texts
def get_tokenizer(texts):    
    # Pass texts to a Tokenizer to get sequences and pad them    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    
    # Return tokenizer
    return tokenizer
    
    
# Reformat data to be a dictionary with text sequences    
def get_data_dict(tok, texts, basic_data, max_seq_len):
    # Use tokenizer to produce text sequences
    seqs = tok.texts_to_sequences(texts)
    seqs = pad_sequences(seqs, max_seq_len)    
    
    
    # Change data to dictionary format (ID pointing to sequence and label)
    data_dict = {}
    for i in range(len(basic_data)):
        data_dict[basic_data[i][0]] = [seqs[i], basic_data[i][2]]    
    
    
    # Return final data dictionary
    return data_dict

    
# Adds actual image pixels to dataset (as opposed to a reference to file location)
def add_actual_images_to_dataset(raw_data_dict, img_dir, resize_shape):
    # Check for existence of image directory
    if not os.path.isdir(img_dir):
        raise ValueError("The expected image directory does not exist in the GIT repo.")


    # Get images        
    data = []
    for sub_dir in os.listdir(img_dir):
        total_sub_dir = os.path.join(img_dir, sub_dir)
        
        for img_file in os.listdir(total_sub_dir):
            # Get image and convert to RGB
            img_path = os.path.join(total_sub_dir, img_file)
            
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = img.resize(resize_shape)
            img = np.array(img)
                        
            # Format data dict to be (filename: [img, sentence_seq, label])
            orig_id = img_file[img_file.find('-') + 1:img_file.rfind('-')]
            updated_sub_list = [ img ] + raw_data_dict[orig_id]
            
            data.append(updated_sub_list)
    
    
    # Return dataset with actual images
    return data
    

# Shuffle and seperate dataset
def shuffle_and_seperate_data(final_data):
    # Define sub-components
    imgs, w_seqs, labels = [], [], []
    
    for img, seq, label in final_data:
        imgs.append(img)
        w_seqs.append(seq)
        labels.append(label)
    
    
    # Convert into numpy arrays
    imgs = np.array(imgs, dtype=np.float32)
    w_seqs = np.array(w_seqs, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    
    
    # Shuffle data order before returning it
    num_ex = len(imgs)
    perm = np.random.permutation(num_ex)
        
    imgs = imgs[perm]
    w_seqs = w_seqs[perm]
    labels = labels[perm]
    
    
    # Transform labels as one-hot vector
    one_hot_labels = to_categorical(labels, num_classes=2)    
    
    
    # Return tuple of seperated components
    return (imgs, w_seqs, one_hot_labels)
    
    
# Finalizes/formats data from Cornell NLVR dataset
# Returns tokenizer and training/test images, questions (as word sequences), and answers (as Booleans)
def get_data(final_img_dims, max_seq_len):    
    # Ensure that repo is there
    get_cornell_nlvr_repo()

    
    # Get basic data from the JSON files
    train_data = get_basic_data(train_json)
    test_data = get_basic_data(test_json)
    
    
    # Get tokenizer based on texts from training and test data
    def convert_to_correct_words(str):
        return (str.replace('ha ', 'has ').replace('blccks', 
                    'blocks').replace('squere', 'square').replace(
                    'tocuhing', 'touching'))
   
    train_texts = [ convert_to_correct_words(tup[1]) for tup in train_data ]
    test_texts = [ convert_to_correct_words(tup[1]) for tup in test_data ]
    
    combined_texts = train_texts + test_texts    
    tokenizer = get_tokenizer(combined_texts)

    
    # Get data dictionary for training and test sets
    train_data_dict = get_data_dict(tokenizer, train_texts, train_data, max_seq_len)
    test_data_dict = get_data_dict(tokenizer, test_texts, test_data, max_seq_len)
        
        
    # Incorporate actual images with dataset
    resize_shape = (final_img_dims[1], final_img_dims[0])

    final_train_data = add_actual_images_to_dataset(train_data_dict, train_img_dir, resize_shape)
    final_test_data = add_actual_images_to_dataset(test_data_dict, test_img_dir, resize_shape)
            
            
    # Seperate data array into images, word sequences, and labels            
    train_components = shuffle_and_seperate_data(final_train_data)
    test_components = shuffle_and_seperate_data(final_test_data)
    
    
    # Return word_index and data lists
    return tokenizer.word_index, train_components, test_components