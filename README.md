This repo is designed to host functionality for improving,
diversifying, and evaluating DL models in Keras. To date, 
it includes the following:

	-NBA video dataset processing functionality (for use with my 
		'nba_pbp_video_dataset' repository)
	-a variety of methods for Snapshot Ensemble training
	-unique Callbacks for introducing a dynamically 
		increasing Dropout and Stochasitc Depth rates
		as well as one for decreasing a cosine annealing 
		SGD learning rate
	-dataset processing, downloading, and management utilities
	-triplet loss batch generator, loss functions, and train routines
	-NLP tools for getting pre-calculated word vectors
	-Visual Question and Answering (VQA) dataset generators and  
		question augmentation functionality (using synonyms)
