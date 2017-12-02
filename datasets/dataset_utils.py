# Import statements
import numpy as np

from keras import backend as K

from nltk.corpus import comtrans
from nltk.corpus import brown



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

    

####### More Text Processing Functions (mostly for NLTK datasets)    

valid_general_corpuses=['brown']


# Validate languages
def validate_langs_and_get_fileids(langs):
    # Ensure valid input
    lang_opts = ['EN-FR', 'DE-EN', 'DE-FR']
    
    if type(langs) not in (list, tuple):
        raise ValueError("'Langs' variable must be either a tuple or list of desired "
                            "language pairs.")
        
    if len(langs) <= 0 or len(langs) > len(lang_opts):
        raise ValueError("'Langs' variable must be positive and no longer than %d desired "
                            "language pairs." % len(langs))
    
    if len(set(langs)) != len(set(langs)):
        raise ValueError("'Langs' variable cannot have duplicate entries.")
    
    for l_pair in langs:
        if l_pair not in lang_opts:
            raise ValueError("Language translation pair strings must be from the following list:\n"
                                "\t%s" % ', '.join(lang_opts))
        
        
    # Get file ID's based on user desired language    
    fileids=[]
    for lang_file in comtrans.fileids():
        for l_pair in langs:
            if l_pair.lower() in lang_file.lower():
                fileids.append(lang_file)
    

    # Return file IDs
    return fileids
        
        
# Get corpus from key
def get_corpus_handle(corpus):
    if corpus not in valid_general_corpuses:
        raise ValueError("The 'corpus' value must be from the following list:\n"
                                "\t%s" % ', '.join(valid_general_corpuses))

    if corpus == 'brown':
        return brown
    else:
        raise ValueError("There does not yet seem to be an implementation "
                            "for the '%s' corpus yet." % corpus)

                            
# Get corpus categories
def get_corpus_categories(corpus):
    return get_corpus_handle(corpus).categories()


# Get translated corpus stats
def get_translated_corpus_stats(langs=['EN-FR']):
    # Validate languages and get corresponding ID's
    fileids = validate_langs_and_get_fileids(langs)    

    # Get aligned sentences            
    align_sents = comtrans.aligned_sents(fileids=fileids)
    

    # Get character-level parallel corpus
    sources = []
    targets = []
   
    for a_sent in align_sents:
        tmp_src = [ord(ch) for ch in ' '.join(a_sent.words)]
        tmp_tgt = [ord(ch) for ch in ' '.join(a_sent.mots)] 
        
        sources.append(tmp_src)
        targets.append(tmp_tgt)
        
    
    # Get lengths
    src_lens = []
    tgt_lens = []
    for src, tgt in zip(sources, targets):
        src_lens.append(len(src))
        tgt_lens.append(len(tgt))
        
        
    # Return numpy arrays stats
    np_src = np.array(src_lens)
    src_stats = [ np.median(np_src), np.mean(np_src), np.std(np_src) ]
    
    np_tgt = np.array(tgt_lens)
    tgt_stats = [ np.median(np_tgt), np.mean(np_tgt), np.std(np_tgt) ]
    
    return src_stats, tgt_stats
    
    
# Get normal corpus stats
def get_general_corpus_stats(corpus='brown', categories=None, use_paragraphs=False):
    # Verify that categories are valid if they were passed to the function
    if categories is not None:
        poss_categories = get_corpus_categories(corpus)
        for cat in categories:
            if cat not in poss_categories:
                raise ValueError("Invalid category for the '%s' corpus. "
                                    "Must be from the following list:"
                                     "\t%s" % ', '.join(poss_categories))
                       
                       
    # Get corpus handle
    corpus_handle = get_corpus_handle(corpus)
    
    if use_paragraphs:
        tokens = corpus_handle.paras(categories=categories)
    else:
        tokens = corpus_handle.sents(categories=categories)
 
 
    # Generate example_sized text sequences    
    sub_sample_size = 10000
    np_lens = np.ndarray(shape=(sub_sample_size,))

    i = 0
    for tok in tokens.iterate_from(0):
        # Get char list from token (whether paragraph or sentence)
        if use_paragraphs:
            word_list = [ word for sent in tok for word in sent ]
        else:
            word_list = [ word for word in tok ]
        
        char_list = [ ord(ch) for ch in ' '.join(word_list) ]

        # Get length
        np_lens[i] = len(char_list) 
        
        # Break after a sufficient sample taken
        i += 1
        if i == sub_sample_size:
            break
        
        
    # Return basic stats
    return [ np.median(np_lens), np.mean(np_lens), np.std(np_lens) ]
    
    
# Get mapping list and dictionary for encoding chars into sequential values
def get_mapping_items(all_bytes):
    index2char = [0, 1] + all_bytes   # add <EMP>, <EOS> tokens
    char2index = {}
    
    for i, byte in enumerate(index2char):
        char2index[byte] = i
        
    return index2char, char2index
    

# Gets char-level text in various languages from ComTran dataset
def get_translated_corpus_chars(langs=['EN-FR'], min_len=75, max_len=250):
    # Validate languages and get corresponding ID's
    fileids = validate_langs_and_get_fileids(langs)    

                
    # Load desired parallel corpus(es)
    align_sents = comtrans.aligned_sents(fileids=fileids)
    

    # Make character-level parallel corpus
    all_bytes = []
    sources = []
    targets = []
   
    for a_sent in align_sents:
        tmp_src = [ord(ch) for ch in ' '.join(a_sent.words)]
        tmp_tgt = [ord(ch) for ch in ' '.join(a_sent.mots)] 
        
        sources.append(tmp_src)
        targets.append(tmp_tgt)
        
        new_bytes = np.unique(tmp_src + tmp_tgt)
        for b in new_bytes:
            if b not in all_bytes:
                all_bytes.append(b)

        
    # Translate all possible bytes into sequential values
    index2char, char2index = get_mapping_items(all_bytes)
        
        
    # Remove short and long sentences
    src = []
    tgt = []
    
    for s, t in zip(sources, targets):
        if min_len <= len(s) < max_len and min_len <= len(t) < max_len:
            src.append(s)
            tgt.append(t)

            
    # Convert char bytes to encoded list of sequential indices
    for i in range(len(src)):
        src[i] = [char2index[ch] for ch in src[i]]
        tgt[i] = [char2index[ch] for ch in tgt[i]]

        
    # Add <EOS> to end of sentence and then padding to make "max_len" length
    for i in range(len(src)):
        src[i] += [1]
        src[i] += [0] * (max_len - len(src[i]))
        
        tgt[i] += [1]
        tgt[i] += [0] * (max_len - len(tgt[i]))

        
    # Return source, target, and means of converting them back into char values
    return src, tgt, index2char

    
# Generator for char-level text blobs from various corpus (in NLTK framework)
def gen_corpus_examples(example_size, min_example_size=None, 
                            trim_long_examples=True, corpus='brown', 
                            categories=None, use_paragraphs=False):
                        
    # Verify that categories are valid if they were passed to the function
    if categories is not None:
        poss_categories = get_corpus_categories(corpus)
        for cat in categories:
            if cat not in poss_categories:
                raise ValueError("Invalid category for the '%s' corpus. "
                                    "Must be from the following list:"
                                     "\t%s" % ', '.join(poss_categories))
            
            
    # Get tokens from brown corpus        
    corpus_handle = get_corpus_handle(corpus)

    if use_paragraphs:
        tokens = corpus_handle.paras(categories=categories)
    else:
        tokens = corpus_handle.sents(categories=categories)
    
    
    # Generate example_sized text sequences
    if (min_example_size is None or min_example_size > example_size):
        min_example_size = int(0.25 * example_size)
    
    all_bytes = []
    for tok in tokens.iterate_from(0):
        # Get char list from token (whether paragraph or sentence)
        if use_paragraphs:
            word_list = [ word for sent in tok for word in sent ]
        else:
            word_list = [ word for word in tok ]
        
        char_list = [ ord(ch) for ch in ' '.join(word_list) ]
        
        
        # Ensure that the example is not too short or too long
        if len(char_list) < min_example_size:
            continue
        
        if trim_long_examples:
            char_list = char_list[:(example_size-1)]
        elif len(char_list) >= example_size:
            continue
        
        
        # Get new bytes and add to list of unique bytes
        new_bytes = np.unique(char_list)
        for b in new_bytes:
            if b not in all_bytes:
                all_bytes.append(b)
        
        
        # Translate all possible bytes into sequential values
        index2char, char2index = get_mapping_items(all_bytes)
                
        # Convert char bytes to encoded list of sequential indices
        for i in range(len(char_list)):
            final_char_list = [ char2index[ch] for ch in char_list ]
 
 
        # Add <EOS> to end of sequence and then pad to make "example_size" length
        final_char_list += [1]
        final_char_list += [0] * (example_size - len(final_char_list))
            
            
        # Return decoder list and then iterate to next phrase
        yield index2char, final_char_list