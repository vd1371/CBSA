import pandas as pd

from ._drop_content_null import _drop_content_null
from ._drop_non_chinese import _drop_non_chinese

def get_data(**params):
	eng_cols = params.get("eng_cols")
	n_samples = params.get("n_samples")

	print('Trying to get_data...')

	direc = "./Data/analytics_challenge_dataset_ex210911-Clean.csv"
	small_direc = "./Data/analytics_challenge_dataset_ex210911-Small.csv"
	df = pd.read_csv(direc if n_samples == None else small_direc, index_col = 0)

	if not n_samples == None:
		df = df.iloc[:n_samples, :]

	# df.dropna(axis = 0, inplace = True)

	return df