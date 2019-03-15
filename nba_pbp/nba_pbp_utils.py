# Import statements
import os, gc, time
import random, multiprocessing
import numpy as np

import cv2		# pip install opencv-python
from xml.etree import ElementTree
from skvideo.io import FFmpegReader



# Global variables
PRINT_ERRORS=False

BRIGHT_MIN=0.35                        
EXPECTED_LOW_RES=(270, 480, 3)
EXPECTED_HIGH_RES=(432, 768, 3)

MIN_VALID_SET_PERCENTAGE=0.1
MIN_TRAIN_SET_PERCENTAGE=0.5

MIN_TRAINING_GAMES_IN_DATASET=125
MIN_VALIDATION_GAMES_IN_DATASET=50

MIN_FRAMES_P_EVENT=150
MAX_FRAMES_P_EVENT=300

SKIP_FRAMES_EVENT=3
SKIP_FRAMES_NO_EVENT=15

MAX_THREADS=64
NUM_BATCH_BEFORE_LOAD=3

LABEL_INDEX=0
JERSEY_INDEX=1
TIME_INDEX=2
VALID_OUTPUT_OPTS=['events', 'jersey_nums', 'time']

GAME_DONE_FILE='check_complete'  # Should be present due to 'remove_partial_downloads.bash'
EVENT_FILE='event.xml'
HIGH_RES_FILE='high_res.mp4'
LOW_RES_FILE='low_res.mp4'
DONE_FILE='done'  
VALID_EVENT_FILES=[EVENT_FILE, HIGH_RES_FILE, LOW_RES_FILE, DONE_FILE]

EVENT_FREQ = { 'low': 0.05, 'med': 0.15, 'high': 0.8 }
HIGH_FREQ_EVENTS=['2PT', '2PTASSIST', 'FOUL', 'FGA', 'REBOUND']
MED_FREQ_EVENTS=['TURNOVER', 'FGABLOCK', 'TURNOVERSTEAL', '3PTASSIST']
LOW_FREQ_EVENTS=['FTMISS', 'TIP', '2PTREBOUND', 'FGAREBOUND', 
                    'FTMAKE', '3PT', 'FOULTURNOVER']

MAX_EVENTS_PER_VID=2
MAX_JERSEY_POSSIBILITIES=((MAX_EVENTS_PER_VID * 100) + 1)

TIME_STAMP_DIGITS=6
POSS_PER_DIGIT=11         



# Helper functions    
def which(program):
    def is_executable(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_executable(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exec_file = os.path.join(path, program)
            if is_executable(exec_file):
                return exec_file

    return None

	
def get_subdirs(parent_dir):
    return [ parent_dir + "/" + name for name in os.listdir(parent_dir)
                                if os.path.isdir(os.path.join(parent_dir, name)) ]

								
def adj_brightness(image, brightness_adj_val):
    # Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
	# Generate new random brightness
    hsv[:, :, 2] = brightness_adj_val * hsv[:, :, 2]
	
	# Convert back to RGB colorspace
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	
def hflip_img(image):
    return cv2.flip(image, 0)

	
								
# Useful utility functionality specifically for building a NBA PBP generator								
def get_game_dirs(root_database_dir):
        game_dirs = [ cur_game_dir for cur_game_dir in get_subdirs(root_database_dir) 
                            if os.path.isfile(cur_game_dir + "/" + GAME_DONE_FILE) ]
        
        random.shuffle(game_dirs)
        return game_dirs

        
def get_game_event_dirs(game_id_dir):
    game_id_events = get_subdirs(game_id_dir)                            
    random.shuffle(game_id_events)

    return game_id_events
   
   
def get_xml_info(label_xml_path):
    event_labels = []
    jersey_nums = []
    time_vals = []
    
    # Use XML to return each label item in its own item
    try:
        xml_obj = ElementTree.parse(label_xml_path).getroot()
    
        for dict_el in xml_obj.iterfind('label'):
            tmp_time_vals = []
            color = 0
            
            for el in dict_el:
                if el.tag == 'title':                        
                    event_labels.append(el.text)
                    
                elif el.tag == 'jersey':
                    jersey_nums.append(int(el.text) + (100 * color))
                    
                elif el.tag == 'color':
                    if el.text == 'AWAY':
                        color = 1
                        
                elif el.tag == 'minutes':
                    min = int(el.text)
                    tmp_time_vals.append(min // 10)
                    tmp_time_vals.append(min % 10)

                elif el.tag == 'seconds':
                    epsilon = 0.001
                    float_secs = float(el.text)                            
                    secs = int(float_secs)
                    decimal_val = int(float((float_secs - secs + epsilon) / 0.1))
                    
                    tmp_time_vals.append(secs // 10)
                    tmp_time_vals.append(secs % 10)
                    tmp_time_vals.append(decimal_val)
                    
                else:
                    tmp_time_vals.append(int(el.text))

            time_vals.append(tmp_time_vals)
    except:
        return None
   
    # Returned as (event string, jersey list (2 x 201), and time digit list (2 x 6 x 11))
    return (("".join(event_labels)), jersey_nums, time_vals)
   

def get_mp4_frames(mp4_path, skip_frames, num_frames_per_event, 
						do_flip, brighten_val, is_high_res, do_aug):
    
    # Get mp4 reader
    try:
        reader = FFmpegReader(mp4_path)     
    except Exception as e:
        if PRINT_ERRORS:
            print(e)
			
        return None
        
    # Get starting frame and offsets
    frame_shape = EXPECTED_HIGH_RES if is_high_res else EXPECTED_LOW_RES            
    start_frame = (reader.inputframenum - (num_frames_per_event * skip_frames)) // 2
    
    if start_frame <= 0:
        reader.close()
        return None

    start_x = int((frame_shape[0] - reader.outputheight) // 2)
    if start_x < 0:
        reader.close()
        return None
        
    start_y = int((frame_shape[1] - reader.outputwidth) // 2)
    if start_y < 0:
        reader.close()
        return None
        
    start_z = int((frame_shape[2] - reader.outputdepth) // 2)    
    if start_z < 0:
        reader.close()
        return None
        
    # Put middle (num_frames_per_event * skip_frames) input frames in numpy array
    cur_i = 0
    cur_frame = 0                
    
    frame_array = np.zeros(shape=((num_frames_per_event, ) + 
                                        frame_shape), dtype=np.uint8)
    
    for frame in reader.nextFrame():
        if cur_frame >= start_frame:    
            cur_offset = cur_frame - start_frame
            if cur_i < num_frames_per_event and (cur_offset % skip_frames) == 0:
                frame_array[cur_i, 
                                start_x:start_x+reader.outputheight, 
                                start_y:start_y+reader.outputwidth,
                                start_z:start_z+reader.outputdepth] = frame
				
                if brighten_val < 1.0:
				    frame_array[cur_i, :, :, :] = adj_brightness(frame_array[cur_i, :, :, :], brighten_val)
                                                                        
                if do_flip:
                    frame_array[cur_i, :, :, :] = hflip_img(frame_array[cur_i, :, :, :])                    
                    
                cur_i += 1
                
        cur_frame += 1
        
    reader.close()    
        
    # Return array with frames
    return frame_array

    
def event_str_to_one_hot(event_label, poss_event_list):
    label_i = poss_event_list.index(event_label)
    label_arr = np.zeros(shape=(1, len(poss_event_list), ), dtype=np.uint8)
    label_arr[0][label_i] = 1
    
    return label_arr
    
    
def jersey_nums_to_one_hot(jersey_nums):
    jersey_arr = np.zeros(shape=(1, MAX_EVENTS_PER_VID, 
                                MAX_JERSEY_POSSIBILITIES), dtype=np.uint8)
                                
    for i, jnum in enumerate(jersey_nums):
        jersey_arr[0][i][jnum] = 1
    
    for i in range(len(jersey_nums), MAX_EVENTS_PER_VID):
        jersey_arr[0][i][MAX_JERSEY_POSSIBILITIES - 1] = 1

    return jersey_arr

    
def time_digits_to_one_hot(time_vals):
    time_arr = np.zeros(shape=(1, MAX_EVENTS_PER_VID, TIME_STAMP_DIGITS, 
                                    POSS_PER_DIGIT), dtype=np.uint8)
                                
    for i in range(len(time_vals)):
        for j in range(TIME_STAMP_DIGITS):
            time_arr[0][i][j][time_vals[i][j]] = 1
            
    for i in range(len(time_vals), MAX_EVENTS_PER_VID):
        for j in range(TIME_STAMP_DIGITS):                        
            time_arr[0][i][j][POSS_PER_DIGIT - 1] = 1
    
    return time_arr
    
    
    
# Generator code                                
'''             
    Overview:
        This generator is designed to work in concert with my NBA play-by-play
        dataset (available as the 'nba_pbp_video_dataset' repo on my GitHub page).
        Accordingly, it is required that before using this generator, you begin 
        downloading the NBA dataset.  Once that dataset's download has progressed 
        sufficiently (e.g. to have completed 75 NBA games), you can then use this 
        generator to feed the data to a deep learning model.  It should be noted 
        that while the generator may not be compitable with every deep learning 
        learning framework out-of-the-box, it has been tested to work with Keras 
        and can easily be modified for use with other frameworks.
        
    Arguments:
        -event_files_dir
            -high level directory location of actual event files
            -will normally be something like '<nba_pbp_repo_dir>/game_events'
        -batch_size
            -integer representing size of each batch
            -this value will have large impact on memory consumption of generator
        -event_types
            -list or tuple with any combination of the 'high', 'med', and/or 'low' strings
            -selects the list(s) from above to use in generating data
            -it is recommended that the following rules are taken into consideration:
                -only one or two of the event_types be used at the same time
                -not to use 'low' and 'high' frequency events at the same time
                -if two of the event_types are used and 'events' are in the output_set, then 
                    the 'keep_cur_balance_factor' can be raised if the generator is going too slowly
        -keep_cur_balance_factor
            -float between 0 and 1 designed to force a balanced mixture of various event types
                -values closer to 0 for even balance of classes
                -values closer to 1 for dataset's natural balance of classes
            -higher values will inherently cause data to generated faster (due to less skipping)
        -event_level_input
            -boolean representing whether to output data on the frame or event level
            -if event_level_input is true, fewer video frames will be skipped during processing
        -num_frames_per_event
            -integer signifying how many of the middle frames of the video to use as input
            -should be no lower than 200 and no higher than 300
        -use_high_res
            -boolean indicating whether to use low or high resolution video
            -note that high resolution videos will consume significantly more memory
        -output_set
            -list or tuple with any combination of the 'events', 'jersey_nums', and/or 'time_remaining' strings
            -will dictate the types of output labels produced by the generator
        -is_validation
            -boolean indicating whether to use training or validatoin set
            -by default it is set to False so it returns the training set generator
        -validation_split
            -float between MIN_TRAIN_SET_PERCENTAGE and 1.0, inclusive
            -default value of 1.0 indicates the training set generator uses the entire <event_files_dir> directory
        -use_video_aug
            -boolean indicating whether to augment videos with hortizontal flipping and brightness adjustment
            -due to large size of entire dataset, this is probably not needed
        -queue_size
            -integer representing the maximum number of batches for queue to hold
            -will block when attempting to add more items to the queue
            -therefore limits maximum memory consumption of each process filling the queue
        -nthread
            -integer representing number of worker threads
            -like with 'batch_size' and 'queue_size', this value should not be adjusted to system's memeory resources
'''

DEFAULT_EXTRA_ARGS = { 'event_types': ['high', 'med'], 'keep_cur_balance_factor': 0.3, 
							'event_level_input': True, 'num_frames_per_event': 250, 
							'use_high_res': False, 'output_set': ['events'], 
							'is_validation': False, 'validation_split': 1.0,
							'use_video_aug': False, 'queue_size': 24, 'nthreads': 2 }


# Look at DEFAULT_EXTRA_ARGS (above) for pre-initialized parameters for the initializer							
class multi_thread_nba_pbp_gen(object):
    def __init__(self, event_files_dir, batch_size, **kwargs):

		# Get other defined parameters
        if kwargs is None:
		    kwargs = DEFAULT_EXTRA_ARGS
        else:
            kwargs = dict(DEFAULT_EXTRA_ARGS.items() + kwargs.items())
			
			
        # Set-up presistent global/shared variables        
        self.keep_cur_balance_factor = kwargs['keep_cur_balance_factor']
        self.out_options = [ False ] * len(VALID_OUTPUT_OPTS)
        
        self.thread_list = []
        self.batch_size = batch_size
        self.nthreads = kwargs['nthreads']
        
        self.event_level_input = kwargs['event_level_input']
        self.num_frames_per_event = kwargs['num_frames_per_event']
        self.use_high_res = kwargs['use_high_res']
        self.use_video_aug = kwargs['use_video_aug']
		
        self.q = multiprocessing.Queue(maxsize=kwargs['queue_size'])	                
        self.manager = multiprocessing.Manager()
        self.remaining_game_lock = multiprocessing.Lock()
        self.remaining_game_list = self.manager.list()  

       
        # Ensure 'ffmpeg' and 'ffprobe' are available	   
        if which("ffmpeg") is None:
            raise ValueError("The directory containing the 'ffmpeg' utility must be on the system's PATH.") 
			
        if which("ffprobe") is None:
            raise ValueError("The directory containing the 'ffprobe' utility must be on the system's PATH.") 
			
   
        # Check event_types argument and set up events list
        self.poss_event_list = []
        if not isinstance(kwargs['event_types'], (list, tuple)):
            raise ValueError('The <event_types> parameter must be a list or tuple.') 

        if len(kwargs['event_types']) < 0:
            raise ValueError('The <event_types> list cannot be empty.') 
            
        if len(kwargs['event_types']) > 3:
            raise ValueError('The <event_types> list has too many options.') 
        
        for e_type in kwargs['event_types']:
            if e_type == 'high':
                self.poss_event_list.extend(HIGH_FREQ_EVENTS)
            elif e_type == 'med':
                self.poss_event_list.extend(MED_FREQ_EVENTS)
            elif e_type == 'low':
                self.poss_event_list.extend(LOW_FREQ_EVENTS)
            else:
                raise ValueError('Invalid output type string provided in <event_types> list.') 
                
                
        # Check number of frames and threads variables
        if self.num_frames_per_event < MIN_FRAMES_P_EVENT:
            raise ValueError('The <num_frames_per_event> value is too small.') 
            
        if self.num_frames_per_event > MAX_FRAMES_P_EVENT:
            raise ValueError('The <num_frames_per_event> value is too large.') 
        
        if self.nthreads > MAX_THREADS or self.nthreads < 1:
            raise ValueError('The <nthreads> value is invalid.') 
        
        if self.event_level_input:
            self.num_frames_per_event = int(self.num_frames_per_event // SKIP_FRAMES_EVENT)
        else:
            self.num_frames_per_event = int(self.num_frames_per_event // SKIP_FRAMES_NO_EVENT)
            
            
        # Use 'game_events' directory for completed games list
        random.seed(os.getpid())	# DO NOT DELETE - REQUIRED FOR 'GET_GAME_DIRS' CALL BELOW	

        if not os.path.isdir(event_files_dir):
            raise ValueError('The string for <event_files_dir> does not exist.') 

        self.subdirs = get_game_dirs(event_files_dir)
      
      
        # Divide into training and validation sets if necessary
        epsilon=0.0001
        if kwargs['is_validation']:
            split_sets_percent = 1.0 - kwargs['validation_split']
            min_percent = MIN_VALID_SET_PERCENTAGE
        else:
            split_sets_percent = kwargs['validation_split']
            min_percent = 1.0 - MIN_TRAIN_SET_PERCENTAGE
        
        if split_sets_percent > 1.0:
            raise ValueError("The 'validation_split' parameter must be at most 1.0.") 
			
        if split_sets_percent + epsilon < min_percent:
            raise ValueError("The 'validation_split' parameter must be between %f and %f." % 
                                (MIN_TRAIN_SET_PERCENTAGE, 1 - MIN_VALID_SET_PERCENTAGE))
			
        split_i = int(kwargs['validation_split'] * len(self.subdirs))
        if kwargs['is_validation']:
            self.subdirs = self.subdirs[split_i:]
            final_min = MIN_VALIDATION_GAMES_IN_DATASET                        			
        else:
            self.subdirs = self.subdirs[:split_i]
            final_min = MIN_TRAINING_GAMES_IN_DATASET

        if len(self.subdirs) <= final_min:
            raise ValueError('Not enough completed games in the <event_files_dir> database directory.') 
				
        for cur_dir in self.subdirs:
            self.remaining_game_list.append(cur_dir)

            
	    # Adjust cur_balance factor to be larger if validation set is small
        if kwargs['is_validation']:
            add_on_val = 1.0 - self.keep_cur_balance_factor
            add_on_val = add_on_val / (1.5 + (float(len(self.subdirs)) / (3 * final_min)))
            
            self.keep_cur_balance_factor += add_on_val

            
        # Check output options
        if not isinstance(kwargs['output_set'], (list, tuple)):
            raise ValueError('The <output_set> parameter must be a list or tuple.') 

        if len(kwargs['output_set']) < 0:
            raise ValueError('The <output_set> list cannot be empty.') 
            
        if len(kwargs['output_set']) > len(VALID_OUTPUT_OPTS):
            raise ValueError('The <output_set> list has too many options.') 
        
        for output_type in kwargs['output_set']:  
            if output_type == VALID_OUTPUT_OPTS[LABEL_INDEX]:
                self.out_options[LABEL_INDEX] = True
            elif output_type == VALID_OUTPUT_OPTS[JERSEY_INDEX]:
                self.out_options[JERSEY_INDEX] = True
            elif output_type == VALID_OUTPUT_OPTS[TIME_INDEX]:
                self.out_options[TIME_INDEX] = True
            else:
                raise ValueError('Invalid output type string provided in <output_set> list.') 
                    
        low_and_high = ('low' in kwargs['event_types'] and 'high' in kwargs['event_types'])
        if (self.out_options[LABEL_INDEX] and low_and_high and 
                    self.keep_cur_balance_factor < 0.95):
            raise ValueError("The 'keep_cur_balance_factor' parameter cannot be lower than 0.95 " +
                                    "if you want to use low and high frequency events together.")

                                    
        # Main functionality loop
        def worker():
            random.seed(os.getpid())
            orig_game_list = self.subdirs
            
            event_vids = []
            event_labels = []        
            label_dict = {}
            
            final_balance_factor = float(1./len(self.poss_event_list))
            final_balance_factor += ((1. - final_balance_factor) * self.keep_cur_balance_factor)
            max_videos_p_event = (int(self.batch_size * NUM_BATCH_BEFORE_LOAD * 
                                            final_balance_factor) + 1)
            
            min_list_size = self.batch_size * NUM_BATCH_BEFORE_LOAD
            if not self.event_level_input:
                min_list_size *= self.num_frames_per_event

                
            while True:
                # Continually try to process new game
                time.sleep(2 * self.nthreads)
                
                with self.remaining_game_lock:
                    if len(self.remaining_game_list) == 0:
                        random.shuffle(orig_game_list)
                        for tmp_dir in orig_game_list:
                            self.remaining_game_list.append(tmp_dir)
                    
                    game_dir = self.remaining_game_list.pop() 
                    
                # Get set of events and process them
                game_events = get_game_event_dirs(game_dir)
                while len(game_events) > 0:
                    # Pick a finished event
                    cur_event_dir = game_events.pop()
                    
                    unfinished=False
                    for file_name in VALID_EVENT_FILES:
                        if not os.path.isfile(cur_event_dir + "/" + file_name):
                            unfinished=True
                            break
                            
                    if unfinished:
                        continue

                        
                    # Process valid event XML file for labels
                    label_xml_path = cur_event_dir + "/" + EVENT_FILE
                    
                    ret_obj = get_xml_info(label_xml_path)
                    if ret_obj is None:
                        continue
                        
                    event_label, jersey_nums, time_vals = ret_obj
                                
                                
                    # Ensure that event is a valid and has not already reached limit                    
                    if event_label not in self.poss_event_list:
                        continue
                    
                    if self.out_options[LABEL_INDEX]:
                        if event_label in label_dict:
                            if label_dict[event_label] <= max_videos_p_event:
                                label_dict[event_label] += 1
                            else:
                                continue
                        else:
                            label_dict[event_label] = long(1)
                    
                    
                   # Get input video frames from mp4      
                    mp4_path = cur_event_dir + "/"
                    if self.use_high_res:
                        mp4_path += HIGH_RES_FILE
                    else:
                        mp4_path += LOW_RES_FILE

                    skip_val = SKIP_FRAMES_EVENT if self.event_level_input else SKIP_FRAMES_NO_EVENT                                                                
                    
                    if self.use_video_aug:
                        rand_flip = int(1000 * random.uniform(0.0, 1.0))
                        rand_flip = True if rand_flip % 2 == 0 else False
                        rand_bright = random.uniform(BRIGHT_MIN, 1.0)						
                    else:
                        rand_flip = False
                        rand_bright = 1.0											
					
                    frame_array = get_mp4_frames(mp4_path, skip_val, self.num_frames_per_event,
													rand_flip, rand_bright, self.use_high_res, 
                                                    self.use_video_aug)
                    
                    if frame_array is None:
                        label_dict[event_label] -= 1
                        continue
                    

                    # Add to running set of inputs and labels until ready to release to queue       
                    cur_event_labels = []
                    
                    # Add inputs
                    if self.event_level_input:
                        event_vids.append(np.expand_dims(frame_array, axis=0))
                    else:
                        for i in range(self.num_frames_per_event):
                            event_vids.append(np.expand_dims(frame_array[i], axis=0))
                        
                    # Get final event label numpy array
                    if self.out_options[LABEL_INDEX]:
                        cur_event_labels.append(event_str_to_one_hot(event_label, self.poss_event_list))
                        
                    # Get final jersey_arr numpy array
                    if self.out_options[JERSEY_INDEX]:
                        cur_event_labels.append(jersey_nums_to_one_hot(jersey_nums))
                        
                    # Get final time numpy array
                    if self.out_options[TIME_INDEX]:
                        cur_event_labels.append(time_digits_to_one_hot(time_vals))
                    
                    # Add the overall label set to the events
                    if self.event_level_input:
                        event_labels.append(cur_event_labels)
                    else:
                        event_labels.extend([ cur_event_labels ] * self.num_frames_per_event)
                        
                        
                    # Release videos/labels if min batch size was reached
                    num_el = len(event_labels)
                    if num_el >= min_list_size:
                        # First shuffle event list
                        c = list(zip(event_vids, event_labels))
                        random.shuffle(c)
                        event_vids, event_labels = zip(*c)
                        
                        # Get batch by batch until not enough elements in the list
                        for start_i in range(0, num_el, self.batch_size):
                            if start_i + self.batch_size > num_el:
                                break
                                                        
                            num_labels = len(event_labels[0])
                            final_labels = [ [] for _ in range(num_labels) ]
                            
                            for label_list in event_labels[start_i:start_i + self.batch_size]:
                                for i in range(num_labels):
                                    final_labels[i].append(label_list[i])
                                        
                            video_batch = np.concatenate(event_vids[start_i:start_i + self.batch_size])
                            labels_batch_list = [ np.concatenate(label_list) for label_list in final_labels ]
                                
                            self.q.put(([video_batch], labels_batch_list))

                        # Reset lists
                        event_labels = []
                        event_vids = []
                        label_dict = {}
                        
                        gc.collect()
                        
                        
        # Start "nthreads" processes to concurrently yield batches
        for _ in range(self.nthreads):
            t = multiprocessing.Process(target=worker)
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

     
    # Functions to get input and output shapes
    def get_input_shape(self):
        if self.event_level_input:
            ret_tup = (self.batch_size, self.num_frames_per_event) 
        else:
            ret_tup = (self.batch_size, )
            
        if self.use_high_res:
            ret_tup += EXPECTED_HIGH_RES
        else:
            ret_tup += EXPECTED_LOW_RES            

        return ret_tup
            
    def get_output_shape(self):
        return (self.batch_size, len(self.poss_event_list))
        
        
    # Function to get a rough estimate for steps per epoch for a generator
    def get_steps_p_epoch(self):
        # Get initial estimate of number of events    		
        tmp_est_num_of_events = len(self.subdirs) * 200
        
        if not self.event_level_input:
            tmp_est_num_of_events *= self.num_frames_per_event		
		
        # Account for select label frequencies
        low_selected = 1 if LOW_FREQ_EVENTS[0] in self.poss_event_list else 0
        med_selected = 1 if MED_FREQ_EVENTS[0] in self.poss_event_list else 0
        high_selected = 1 if HIGH_FREQ_EVENTS[0] in self.poss_event_list else 0
        
        est_num_of_events = 0
        if low_selected:
            est_num_of_events += EVENT_FREQ['low'] * tmp_est_num_of_events
            
        if med_selected:
            est_num_of_events += EVENT_FREQ['med'] * tmp_est_num_of_events
            
        if high_selected:
            est_num_of_events += EVENT_FREQ['high'] * tmp_est_num_of_events
            
		# Return est_num_of_events if label balancing is not a factor
        if not self.out_options[LABEL_INDEX]:
			return int(est_num_of_events / self.batch_size)
		
        # Otherwise return estimated subset of dataset accessed due to balancing
        divide_factor = 1.0
        if low_selected and high_selected:
            return int(est_num_of_events / (3 * self.batch_size))

        one_selected = (low_selected ^ med_selected ^ high_selected)        
        if self.keep_cur_balance_factor < 0.75:
            if self.keep_cur_balance_factor > 0.5:
                if one_selected:
                    divide_factor = 1.25
                else:
                    divide_factor = 2.0
					
            elif self.keep_cur_balance_factor > 0.25:
                if one_selected:
                    divide_factor = 1.5
                else:
                    divide_factor = 3.0
					
            else:
                if one_selected:
                    divide_factor = 2.0
                else:
                    divide_factor = 4.5
		
		# Return adjusted number of steps
        return int(est_num_of_events / (divide_factor * self.batch_size))


    # Stop the generator altogether (e.g. destructor) at the end of use
    def stop_all_threads(self):
        for thread in self.thread_list:
            if thread.is_alive():
                thread.terminate()
            
        self.q.close()			

		
# Function to get complementing training and validation set NBA PBP generators
# Uses the exact same parameters as the 'multi_thread_nba_pbp_gen' generator above
def get_train_val_nba_pbp_gens(event_files_dir, batch_size, 
								validation_split, **kwargs):
    
    # Get training generator
    train_gen = multi_thread_nba_pbp_gen(event_files_dir, batch_size, 
											validation_split=validation_split, 
											is_validation=False, **kwargs)	
	
    # Get validation generator
    if 'nthreads' in kwargs and kwargs['nthreads'] > 2:
        kwargs['nthreads'] = int(kwargs['nthreads'] * 0.7)
        
    val_gen = multi_thread_nba_pbp_gen(event_files_dir, batch_size,
											validation_split=validation_split, 
											is_validation=True, **kwargs)	
		
    # Return both as a tuple    
    return (train_gen, val_gen)


	

    
############ MAIN
'''
print("Hello world.")
mama = multi_thread_nba_pbp_gen('/mnt/efs/pbp_dataset/game_events', 8, 
                                    nthreads=4, event_level_input=True,
                                    keep_cur_balance_factor=.75, validation_split=0.85,
                                    use_high_res=True, use_video_aug=True)

print(mama.get_steps_p_epoch())                                    
for i in range(mama.get_steps_p_epoch()):                 
    whats_next = mama.next()
    if i % 25 == 0:
        print(i)
        
print(type(whats_next))
print(type(whats_next[0]))
print(whats_next[0][0].shape)
print(whats_next[1][0].shape)
print(whats_next[1][0])
quit()
'''