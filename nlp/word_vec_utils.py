# Import statements
import os 
import math
import urllib

import numpy as np
from sklearn.decomposition import PCA
from keras.layers import Embedding

'''
To do:

-add script in "vqa_models" directory to get current "soa_cnns" and "dl_utilities"
    -then check dir into GIT

'''


# Gets shape of word embeddings        
def get_num_words_and_dimensions(w_vector_fname):
    dim = -1
    is_first = True
    
    with open(w_vector_fname) as f:
        for i, l in enumerate(f):
            if dim == -1 and not is_first:
                dim = len(l.split()) - 1
                
            is_first = False
    
    if dim == -1:
        raise ValueError("File has invalid format; it should be in 'fasttext' format.")
    
    return i, dim
        
        
        
def get_pretrained_word_vectors(final_embedding_len, top_percent_of_words=1.0):
    # Get original 300-dimension words embeddings
    cur_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    word_vec_filename = cur_dir + "english_words.vec"
    
    vec_file_URL = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec"

    if not os.path.exists(word_vec_filename):
        print("Getting Facebook's pre-trained word vectors...")
        testfile = urllib.URLopener()
        testfile.retrieve(vec_file_URL, word_vec_filename)
        print("Done.")

    
    # Ensure that new embedding dimensions are usable
    embed_shape = get_num_words_and_dimensions(word_vec_filename)
    
    cur_dims = embed_shape[-1]    
    if cur_dims < final_embedding_len:
        raise ValueError("The 'final_embedding_len' value should be less than %d." % 
                            cur_dims)
    
    
    # Parse word vector file
    num_examples = int(embed_shape[0] * top_percent_of_words)

    embeddings_vectors = np.ndarray((num_examples, cur_dims))
    embeddings_words = []
    
    skipped_first = False
    
    f_handle = open(word_vec_filename, 'r')
    for i, line in enumerate(f_handle):
        # Do not process first line
        if not skipped_first:
            skipped_first = True
            continue
        
        # Only process maximum num of examples
        if i > num_examples:
            break

        # Otherwise, get the word embedding
        values = line.split()
        
        word = values[0]
        embeddings_words.append(word)
        
        coefs = np.asarray(values[1:], dtype='float32')        
        embeddings_vectors[i - 1] = coefs

    f_handle.close()

    
    # Resize dimensions
    PCA_obj = PCA(n_components=final_embedding_len)
    embeddings_vectors = PCA_obj.fit_transform(embeddings_vectors)
    
    
    # Create embedding index
    embeddings_index = {}
    for i, word in enumerate(embeddings_words):
        embeddings_index[word] = embeddings_vectors[i]
        
        
    # Return final embedding index
    return embeddings_index        


    
# Get pre-trained weights and pass through a Keras Embedding layer     
def embed_layer_with_pretrained_word_vecs(tokenizer_index, 
                                            final_embedding_len, sequence_len):    
    # Get pre-trained embeddings
    embeddings_index = get_pretrained_word_vectors(final_embedding_len)
    
    
    # Get final embedding matrix based on current set of words
    num_words = len(tokenizer_index) + 1
    embedding_matrix = np.random.randn(num_words, final_embedding_len)
    embedding_matrix *= float(0.5 / math.sqrt(final_embedding_len))
    
    
    # Get each word embedding it if it exists in pre-trained ones
    for word, i in tokenizer_index.items():
        embedding_vector = embeddings_index.get(word)
        
        # Only set words found in embedding index
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    
    # Return Embedding layer initialized with word vectors weights
    return Embedding(num_words,
                        final_embedding_len,
                        weights=[embedding_matrix],
                        input_length=sequence_len,
                        trainable=False)
    

    