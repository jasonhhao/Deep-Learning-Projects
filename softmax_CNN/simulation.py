# Python Standard Library
import os
import sys
import itertools

# External Dependencies
import numpy as np
import tensorflow as tf; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Deactivates annoying CPU Speed Up Warning 

# Internal Dependencies
import utilities.load_function as load_function
import softmax_cnn
import time


## Define Settings

settings = {
	# Simulation Settings
	'learning_rate': .1,
	'epochs': 500,
	# Architecture Settings
	'filter_dim': 4,
	'filters': 7,
	'dense_h_fc_units': 200,
	'dropout_ratio': .4,
}

path_to_data = 'data/cifar/data_batch_1'
images, labels = load_function.unpickle(os.path.normpath(path_to_data))
classes = np.unique(labels)
data_sets = load_function.load_data()

softmax_results = softmax_cnn.softmax_cnn(data_sets, classes, settings)

print("Done")
