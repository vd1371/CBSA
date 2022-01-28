import time
import json
import gc

from utils import ParallelProcess

def remove_stop_words(X, **params):
	"""
	X: list of list
	"""
	print("removing stop words...")

	stop_words_dir = params.get("stop_words_dir")
	with open("./DataPreprocessing/_text_cleaners/stopwords-zh.json",
				'r', encoding = 'utf-8-sig') as f:
		stop_words = json.load(f)	

	n_cores = params.get("n_cores")

	results = ParallelProcess(X,
								_remove_stop_words_for_parallel,
								stop_words,
								n_cores = n_cores)

	return results

def _remove_stop_words_for_parallel(X,
								stop_words):

	for sw in stop_words:
		for sentence in X:

			if isinstance(sentence, str):
				sentence.replace(sw, "")

			else:
				try:
					while True:
						sentence.remove(sw)
				except ValueError:
					pass

	return X

