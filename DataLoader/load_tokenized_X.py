import json
import ast
import pickle

from .get_data import get_data
from ._get_X_Y_from_df import get_X_Y_from_df

def load_tokenized_X(**params):

	print ("Trying to load_tokenized_X....")

	with open('./Data/TokenizedX.pkl', 'rb') as f:
		X = pickle.load(f)

	return X
