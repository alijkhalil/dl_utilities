# Import statements
import os, gc, time
import random, multiprocessing
import numpy as np

from skvideo.io import FFmpegReader
from xml.etree import ElementTree



# Global variables
MIN_NUM_GAMES_IN_DATASET=75

MIN_FRAMES_P_EVENT=200
MAX_FRAMES_P_EVENT=300

SKIP_FRAMES_EVENT=3
SKIP_FRAMES_NO_EVENT=15

MAX_THREADS=64
NUM_BATCH_BEFORE_LOAD=3

LABEL_INDEX=0
JERSEY_INDEX=1
TIME_INDEX=2
VALID_OUTPUT_OPTS=['events', 'jersey_nums', 'time']

EVENT_FILE='event.xml'
HIGH_RES_FILE='high_res.mp4'
LOW_RES_FILE='low_res.mp4'
DONE_FILE='done'
VALID_EVENT_FILES=[EVENT_FILE, HIGH_RES_FILE, LOW_RES_FILE, DONE_FILE]

HIGH_FREQ_EVENTS=['2PT', '2PTASSIST', 'FOUL', 'FGA', 'REBOUND']
MED_FREQ_EVENTS=['TURNOVER', 'FGABLOCK', 'TURNOVERSTEAL', '3PTASSIST']
LOW_FREQ_EVENTS=['FTMISS', 'TIP', '2PTREBOUND', 'FGAREBOUND', 
                    'FTMAKE', '3PT', 'FOULTURNOVER']

MAX_EVENTS_PER_VID=2
MAX_JERSEY_POSSIBILITIES=((MAX_EVENTS_PER_VID * 100) + 1)

TIME_STAMP_DIGITS=6
POSS_PER_DIGIT=11         



# Helper functions    
def get_subdirs(parent_dir):
    return [ parent_dir + "/" + name for name in os.listdir(parent_dir)
                                if os.path.isdir(os.path.join(parent_dir, name)) ]

                                

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
            -it is recommended that the following rules are taken into consideration
                -only one or two of the event_types be used at the same time
                -not to use 'low' and 'high' frequency events at the same time
                -if two of the event_types are used and 'events' are in the output_set, then 
                    the 'imbalance_factor' can be raised if the generator is going too slowly
        -imbalance_factor
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
        -queue_size
            -integer representing the maximum number of batches for queue to hold
            -will block when attempting to add more items to the queue
            -therefore limits maximum memory consumption of each process filling the queue
        -nthread
            -integer representing number of worker threads
            -like with 'batch_size' and 'queue_size', this value should not be adjusted to system's memeory resources
'''
class multi_thread_nba_pbp_gen(object):
    def __init__(self, event_files_dir, batch_size, event_types=['high', 'med'],
                        imbalance_factor=0.3, event_level_input=True, 
                        num_frames_per_event=250, use_high_res=False, 
                        output_set=['events'], queue_size=16, nthreads=2):	
                    
        # Set-up presistent global/shared variables        
        self.subdirs = []
        self.out_options = [ False ] * len(VALID_OUTPUT_OPTS)
        
        self.thread_list = []
        self.nthreads = nthreads
        self.batch_size = batch_size
        
        self.event_level_input = event_level_input
        self.num_frames_per_event = num_frames_per_event
        self.use_high_res = use_high_res
                
        self.q = multiprocessing.Queue(maxsize=queue_size)	                
        self.manager = multiprocessing.Manager()
        self.remaining_game_lock = multiprocessing.Lock()
        self.remaining_game_list = self.manager.list()  

        
        # Check event_types argument and set up events list
        self.poss_event_list = []
        if not isinstance(event_types, (list, tuple)):
            raise ValueError('The <event_types> parameter must be a list or tuple.') 

        if len(event_types) < 0:
            raise ValueError('The <event_types> list cannot be empty.') 
            
        if len(event_types) > 3:
            raise ValueError('The <event_types> list has too many options.') 
        
        for e_type in event_types:
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
            
            
        # Check provided 'game_events' directory        
        random.seed(os.getpid())

        if not os.path.isdir(event_files_dir):
            raise ValueError('The string for <event_files_dir> does not exist.') 
            
        tmp_game_ids = get_subdirs(event_files_dir)
        self.subdirs.extend(tmp_game_ids)
        self.subdirs = [ cur_game_dir for cur_game_dir in self.subdirs 
                            if os.path.isfile(cur_game_dir + "/done") ]
        
        if len(self.subdirs) <= MIN_NUM_GAMES_IN_DATASET:
            raise ValueError('Not enough sub-directories (representing number of games) in the <event_files_dir>.') 

        random.shuffle(self.subdirs)
        for cur_dir in self.subdirs:
            self.remaining_game_list.append(cur_dir)
            
            
        # Check output options
        if not isinstance(output_set, (list, tuple)):
            raise ValueError('The <output_set> parameter must be a list or tuple.') 

        if len(output_set) < 0:
            raise ValueError('The <output_set> list cannot be empty.') 
            
        if len(output_set) > len(VALID_OUTPUT_OPTS):
            raise ValueError('The <output_set> list has too many options.') 
        
        for output_type in output_set:  
            if output_type == VALID_OUTPUT_OPTS[LABEL_INDEX]:
                self.out_options[LABEL_INDEX] = True
            elif output_type == VALID_OUTPUT_OPTS[JERSEY_INDEX]:
                self.out_options[JERSEY_INDEX] = True
            elif output_type == VALID_OUTPUT_OPTS[TIME_INDEX]:
                self.out_options[TIME_INDEX] = True
            else:
                raise ValueError('Invalid output type string provided in <output_set> list.') 
                
                    
                    
        # Main functionality loop
        def worker():
            random.seed(os.getpid())
            orig_game_list = self.subdirs
            
            event_vids = []
            event_labels = []        
            label_dict = {}
            
            actual_imbalance_factor = float(1./len(self.poss_event_list))
            actual_imbalance_factor += ((1. - actual_imbalance_factor) * imbalance_factor)
            max_videos_p_event = (int(self.batch_size * NUM_BATCH_BEFORE_LOAD * 
                                            actual_imbalance_factor) + 1)
            
            min_list_size = self.batch_size * NUM_BATCH_BEFORE_LOAD
            if not self.event_level_input:
                min_list_size *= self.num_frames_per_event
            
            print(max_videos_p_event)
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
                game_events = get_subdirs(game_dir)                
                random.shuffle(game_events)
                
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
                    event_file_path = cur_event_dir + "/" + EVENT_FILE
                    xml_obj = ElementTree.parse(event_file_path).getroot()

                    final_labels = []
                    jersey_nums = []
                    time_vals = []
                    
                    try:
                        for dict_el in xml_obj.iterfind('label'):
                            tmp_time_vals = []
                            color = 0
                            
                            for el in dict_el:
                                if el.tag == 'title':                        
                                    final_labels.append(el.text)
                                    
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
                        continue
                                
                                
                    # Ensure that event is a valid and needed event label
                    label = "".join(final_labels)
                    if label not in self.poss_event_list:
                        continue
                    
                    if self.out_options[LABEL_INDEX]:
                        if label in label_dict:
                            if label_dict[label] <= max_videos_p_event:
                                label_dict[label] += 1
                            else:
                                continue
                        else:
                            label_dict[label] = 1
                    
                    
                   # Open mp4 reader        
                    mp4_path = cur_event_dir + "/"
                    if self.use_high_res:
                        mp4_path += HIGH_RES_FILE
                    else:
                        mp4_path += LOW_RES_FILE

                    reader = FFmpegReader(mp4_path)     
                        
                        
                    # Get input frames
                    skip_val = SKIP_FRAMES_EVENT if self.event_level_input else SKIP_FRAMES_NO_EVENT                                                            
                    start_frame = (reader.inputframenum - (self.num_frames_per_event * skip_val)) // 2
                    
                    if start_frame <= 0:
                        reader.close()
                        continue

                    cur_i = 0
                    cur_frame = 0                
                    frame_array = np.ndarray(shape=(self.num_frames_per_event, 
                                                    reader.outputheight, 
                                                    reader.outputwidth, 
                                                    reader.outputdepth)).astype(reader.dtype)
                        
                    for frame in reader.nextFrame():
                        if cur_frame >= start_frame:    
                            cur_offset = cur_frame - start_frame
                            if cur_i < self.num_frames_per_event and (cur_offset % skip_val) == 0:
                                frame_array[cur_i, :, :, :] = frame
                                cur_i += 1
                                
                        cur_frame += 1
                        
                    reader.close()                
                    
                    
                    # Add to running set of inputs and labels until ready to release to queue       
                    gc.collect()
                    cur_event_labels = []
                    
                    # Add inputs
                    if self.event_level_input:
                        event_vids.append(np.expand_dims(frame_array, axis=0))
                    else:
                        for i in range(frame_array.shape[0]):
                            event_vids.append(np.expand_dims(frame_array[i], axis=0))
                        
                    # Get final label numpy array
                    if self.out_options[LABEL_INDEX]:
                        label_i = self.poss_event_list.index(label)
                        label_arr = np.zeros(shape=(len(self.poss_event_list), ), dtype=np.uint8)
                        label_arr[label_i] = 1
                        
                        cur_event_labels.append(np.expand_dims(label_arr, axis=0))
                        
                    # Get final jersey_arr numpy array
                    if self.out_options[JERSEY_INDEX]:
                        jersey_arr = np.zeros(shape=(MAX_EVENTS_PER_VID, 
                                                    MAX_JERSEY_POSSIBILITIES), dtype=np.uint8)
                                                    
                        for i, jnum in enumerate(jersey_nums):
                            jersey_arr[i][jnum] = 1
                        
                        for i in range(len(jersey_nums), MAX_EVENTS_PER_VID):
                            jersey_arr[i][MAX_JERSEY_POSSIBILITIES - 1] = 1
                            
                        cur_event_labels.append(np.expand_dims(jersey_arr, axis=0))
                        
                    # Get final time numpy array
                    if self.out_options[TIME_INDEX]:
                        time_arr = np.zeros(shape=(MAX_EVENTS_PER_VID, TIME_STAMP_DIGITS, 
                                                    POSS_PER_DIGIT), dtype=np.uint8)
                                                    
                        for i in range(len(time_vals)):
                            for j in range(TIME_STAMP_DIGITS):
                                time_arr[i][j][time_vals[i][j]] = 1
                                
                        for i in range(len(time_vals), MAX_EVENTS_PER_VID):
                            for j in range(TIME_STAMP_DIGITS):                        
                                time_arr[i][j][POSS_PER_DIGIT - 1] = 1
                        
                        cur_event_labels.append(np.expand_dims(time_arr, axis=0))
                    
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
                        
                        # Get batch by batch until not enough list
                        for start_i in range(0, num_el, self.batch_size):
                            if start_i + self.batch_size > num_el:
                                break
                            
                            video_batch = np.concatenate(event_vids[start_i:start_i + self.batch_size])
                            
                            num_labels = len(event_labels[0])
                            final_labels = [ [] for _ in range(num_labels) ]
                            for label_list in event_labels[start_i:start_i + self.batch_size]:
                                for i in range(num_labels):
                                    final_labels[i].append(label_list[i])
                            
                            batch_labels_list = [ np.concatenate(label_list) 
                                                    for label_list in final_labels ]
                                
                            self.q.put(([video_batch], batch_labels_list))
                            
                        # Reset lists
                        event_labels = []
                        event_vids = []
                        label_dict = {}
                        quit()

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

        
    # Stop the generator altogether (e.g. destructor) at the end of use
    def stop_all_threads(self):
        for thread in self.thread_list:
            if thread.is_alive():
                thread.terminate()
            
        self.q.close()			
        
        
        
        
        
############ MAIN

print("Hello world.")
mama = multi_thread_nba_pbp_gen('/mnt/efs/pbp_dataset/game_events', 30, 
                                    nthreads=2, event_level_input=False,
                                    imbalance_factor=0.0)
whats_next = mama.next()
print(type(whats_next))
print(type(whats_next[0]))
print(whats_next[0][0].shape)
print(whats_next[1][0].shape)
print(whats_next[1][0])
quit()