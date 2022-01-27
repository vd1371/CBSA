import pandas as pd
import numpy as np

def _split_samples_for_processes(inp, process_number, len_sections):

	L = len(inp)
	start_ind = process_number * len_sections
	end_ind = min((process_number+1)*len_sections, L)

	if isinstance(inp, pd.DataFrame):
		samples = inp.iloc[start_ind: end_ind, :]
	elif isinstance(inp, pd.Series):
		samples = inp.iloc[start_ind: end_ind]
	elif isinstance(inp, (list, np.ndarray)):
		samples = inp[start_ind: end_ind]
	else:
		raise TypeError ("Unknown type for inp")

	return samples