import numpy as np
import json
import pickle

from DataLoader import load_tokenized_X

# from utils import count_num_words
# from utils import plot_length
from utils import ParallelProcess

def find_unique_words_and_save_to_json(**params):

	print ("Trying to find_unique_words_and_save_to_json...")
	should_plot_length = params.get("should_plot_length")

	X = load_tokenized_X(**params)
	# print((count_num_words(X)))
	if should_plot_length:
		plot_length(X, **params)

	n_cores = params.get("n_cores")

	results = ParallelProcess(X,
								_find_unique_words_for_parallel,
								n_cores = n_cores)

	unique_words = np.unique(results).tolist()

	with open('./Data/unique.pkl', 'wb') as f:
		pickle.dump(unique_words, f)


def _find_unique_words_for_parallel(X):

	holder = [item for sublist in X for item in sublist]
	return np.unique(holder).tolist()