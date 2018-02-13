# Import statements
import cv2, os, math, random

import numpy as np
import cPickle as pickle

from keras.utils.np_utils import to_categorical



# Global variables
cur_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
gen_data_dir = cur_dir + 'generated_data/'

colors_options = [      (0,0,255), ## r
                        (0,255,0), ## g
                        (255,0,0), ## b
                        (0,156,255), ## o
                        (128,128,128), ## k (grey)
                        (0,255,255) ## y
                    ]
                    
num_colors = len(colors_options)
num_question_subtypes = 3                    

question_dim = 2 * num_colors * num_question_subtypes
anwser_dim = 6   # max 2 for yes/no, max 2 for shapes, max (N - 1) for numner of "other" objects


    
# Helper function to build the artificial dataset
def get_dataset_token(img_side_size, half_shape_size, num_q_per_image): 
    # Define variables
    objects = []
                        
    # Ensure that image will be big enough (to fit 5 times the number of shapes)
    min_size = int(math.sqrt((((half_shape_size * 2) ** 2) * num_colors * 5)))
    if img_side_size < min_size:
        raise ValueError("The image size should be at least %d (per dimension) given "
                            "the current shape size.\nYou should either decrease the "
                            "shape size or increase the image size." % min_size)
   
   
    # Create a square image with an all-black background
    img = np.ones((img_side_size, img_side_size, 3)) * 255
    
    
    # Add a shape for each color option
    for color_id, color in enumerate(colors_options):  
        # Get x/y center coordinates for shape
        while True:
            ok = True
            
            new_center = np.random.randint(half_shape_size, img_side_size - half_shape_size, 2)        
            for _, c, _ in objects:
                if ((new_center - c) ** 2).sum() < ((half_shape_size * 2) ** 2):
                    ok = False
                        
            if ok:
                break
                
        
        # Make shape either a rectangle or circle (half of the time)       
        if random.random() < 0.5:
            top_left_corner = (new_center[0] - half_shape_size, new_center[1] - half_shape_size)
            bottom_right_corner = (new_center[0] + half_shape_size, new_center[1] + half_shape_size)
            
            cv2.rectangle(img, top_left_corner, bottom_right_corner, color, -1)
            objects.append((color_id, new_center, 'r'))
        else:
            center_tuple = (new_center[0], new_center[1])
            
            cv2.circle(img, center_tuple, half_shape_size, color, -1)
            objects.append((color_id, new_center, 'c'))


    # Use images to produce questions and answer 
    non_relational_index = 6    
    relation_index = 7
    subtype_index = 8    

    
    # Non-relational questions
    norel_questions = []
    norel_answers = []
    
    for _ in range(num_q_per_image):
        # Get question
        question = np.zeros((question_dim))
        
        color = random.randint(0,num_colors-1)        
        subtype = random.randint(0,num_question_subtypes-1)
        
        question_i = 0
        question_i += (subtype * num_colors) + color
        question[question_i] = 1
        
        
        # Answers are encoded so each answer number appears roughly same amount of time
        """Answer possibilities: [yes, no, rectangle, circle, <num_objects>]"""
        
        if subtype == 0:
            """query shape -> rectangle/circle"""
            if objects[color][2] == 'r':
                answer = 0
            else:
                answer = 1

        elif subtype == 1:
            """query horizontal position -> yes/no"""
            if objects[color][1][0] < img_side_size / 2:
                answer = 2
            else:
                answer = 3

        elif subtype == 2:
            """query vertical position -> yes/no"""
            if objects[color][1][1] < img_side_size / 2:
                answer = 4
            else:
                answer = 5
        
        # Add question and answer to non-relational
        norel_questions.append(question)
        norel_answers.append(answer)
    
    
    """Relational questions"""
    rel_answers = []
    rel_questions = []
    
    for _ in range(num_q_per_image):
        # Get question
        question = np.zeros((question_dim))

        color = random.randint(0,num_colors-1)        
        subtype = random.randint(0,num_question_subtypes-1)
        
        question_i = num_question_subtypes * num_colors
        question_i += (subtype * num_colors) + color
        question[question_i] = 1

            
        # Answers are encoded so each answer number appears roughly same amount of time
        """Answer possibilities: [yes, no, rectangle, circle, <num_objects>]"""
        
        if subtype == 0:
            """closest to current -> rectangle/circle"""
            my_obj_center = objects[color][1]
            dist_list = [((my_obj_center - other_center) ** 2).sum() for _, other_center, _ in objects]
            dist_list[dist_list.index(0)] = (img_side_size ** 2)
            closest = dist_list.index(min(dist_list))
            
            if objects[closest][2] == 'r':
                answer = 0
            else:
                answer = 1
                
        elif subtype == 1:
            """furthest from current -> rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))
            
            if objects[furthest][2] == 'r':
                answer = 3
            else:
                answer = 4

        elif subtype == 2:
            """count with the same shape -> 0~5"""
            cur_obj_shape = objects[color][2]
            
            count = -1
            for _, _, obj_shape in objects:
                if obj_shape == cur_obj_shape:
                    count += 1 
            
            answer = count
            
        # Add question and answer to non-relational
        rel_questions.append(question)
        rel_answers.append(answer)

        
    # Format questions/answer as tuples        
    img = img / 255.
    relations = (rel_questions, rel_answers)
    non_relations = (norel_questions, norel_answers)
    
    dataset_item = (img, relations, non_relations)
    
    
    # Return dataset as (img, rel_data, non_relations)
    return dataset_item


# If loading from a pre-generated dataset, none of the parameters are needed
# However, if a new dataset is desired, set the parameters and simply flip the       
#       'force_new' variable.  That dataset will be saved automatically in the 
#       './generated_data/sort-of-clevr.pickle' file.
def get_simple_vqa_artificial_dataset(train_size=5000, test_size=1000, 
                                        num_q_per_image=5, 
                                        img_side_size=75, half_shape_size=5,
                                        force_new=False):
        
    # Ensure that file is there or generate and save a new dataset if required/requested
    data_filename = ('sort_of_clevr_%d_%d_%d_%d_%d.pickle' %
                        (train_size, test_size, num_q_per_image, 
                        img_side_size, half_shape_size))
                        
    dataset_path = os.path.join(gen_data_dir, data_filename)
    if not os.path.exists(dataset_path) or force_new:
        print('Creating and saving a new dataset...')
        
        # Generate dataset
        train_dataset = [ get_dataset_token(img_side_size, half_shape_size, num_q_per_image) 
                                for _ in range(train_size) ]
        test_dataset = [ get_dataset_token(img_side_size, half_shape_size, num_q_per_image) 
                                for _ in range(test_size) ]
                                
                                

        # Save both datasets                            
        if not os.path.isdir(gen_data_dir):
            os.mkdir(gen_data_dir)

        with open(dataset_path, 'wb') as f:
            pickle.dump((train_dataset, test_dataset), f)
                      
    else:
        print('Loading pre-generated data...')

        with open(dataset_path, 'rb') as f:
            train_dataset, test_dataset = pickle.load(f)


    # Groups each training question with its answer
    rel_train = []
    non_rel_train = []                
    
    for img, relations, non_relations in train_dataset:
        for qst, ans in zip(relations[0], relations[1]):
            rel_train.append((img, qst, ans))
            
        for qst, ans in zip(non_relations[0], non_relations[1]):
            non_rel_train.append((img, qst, ans))

            
    # Groups each test question with its answer
    rel_test = []
    non_rel_test = []
    
    for img, relations, non_relations in test_dataset:
        for qst, ans in zip(relations[0], relations[1]):
            rel_test.append((img, qst, ans))
            
        for qst, ans in zip(non_relations[0], non_relations[1]):
            non_rel_test.append((img, qst, ans))
            
            
    # Return training and test datasets
    print("Done.")
    return ((rel_train, non_rel_train), (rel_test, non_rel_test))
    
    
# Reformat dataset tuples to be in img_list, question_list, answer_list (one-hot)
def reformat_dataset(dataset, shuffle=True):
    # Get processing order
    dataset_len = len(dataset)
    
    if shuffle:
        process_order = np.random.permutation(dataset_len)
    else:
        process_order = range(dataset_len)
    
    
    # Separate out images, questions, and answers as lists
    imgs = []
    question_embed = []
    anwsers = []
    
    for i in process_order:
        imgs.append(dataset[i][0])
        question_embed.append(dataset[i][1])
        anwsers.append(dataset[i][2])

        
    # Convert answers to one-hot vector
    one_hot_answers = to_categorical(np.array(anwsers), num_classes=anwser_dim)    
    
    
    # Return lists as numpy arrays   
    return np.array(imgs), np.array(question_embed), one_hot_answers