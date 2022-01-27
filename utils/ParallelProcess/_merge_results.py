import pandas as pd
import numpy as np

import gc

def merge_results(holder):

	holder = sorted(holder, key=lambda x: x[0])
	holder = [val for _, val in holder]

	if isinstance(holder[0], list):
		results = []
		for sample in holder:
			results += sample

	elif isinstance(holder[0], pd.DataFrame):
		results = pd.concat(holder, axis = 0)

	elif isinstance(holder[0], pd.Series):
		results = pd.concat(holder)

	elif isinstance(holder[0], np.ndarray):
		results = np.concatenate(holder)

	gc.collect()

	return results