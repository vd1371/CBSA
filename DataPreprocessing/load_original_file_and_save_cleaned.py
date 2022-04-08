import pandas as pd
import json
import pickle

from DataLoader import get_data
from DataLoader import get_X_Y_from_df
from ._text_cleaners import remove_space_from_all_samples
from ._text_cleaners import remove_stop_words
from ._replace_bad_chars import replace_bad_chars


def load_original_file_and_save_cleaned(**params):

	df = get_data(**params)
	X, _ = get_X_Y_from_df(df, **params)
	X = replace_bad_chars(X, **params)
	X = remove_space_from_all_samples(X)
	X = remove_stop_words(X, **params)

	print ("Trying to pickle the CleanX")
	with open('./Data/CleanX.pkl', 'wb') as f:
		pickle.dump(X, f)

	print ("Clean file is saved.")