This repo is designed to host functionality for improving,
diversifying, and evaluating DL models in Keras. To date, 
it includes the following:

	-a variety of methods for Snapshot Ensemble training
	-unique Callbacks for introducing a dynamically 
		increasing Dropout and Stochasitc Depth rates
		as well as one for decreasing a cosine annealing 
		SGD learning rate
	-dataset processing, downloading, and management utilities
	-triplet loss batch generator, loss functions, and train routines
