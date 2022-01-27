import ast

from .dicts_and_lists import replacements

from utils import ParallelProcess

def _replace_bad_chars(X, **params):

	print ("Trying to replace bad chars")
	repls = replacements()

	n_cores = params.get("n_cores")
	results = ParallelProcess(X,
								_replace_bad_chars_for_parallel,
								repls,
								n_cores = n_cores)

	return results

def _replace_bad_chars_for_parallel(X,
									repls):

	for num, (bad_char, j) in enumerate(repls.items()):
		# print (f"{bad_char} is being replaced ({num+1}/{L})")
		X = X.str.replace(bad_char, j, regex=False)

	return X