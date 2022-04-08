import json
import ast
import pickle

def load_clean_X(**params):

	print ("Trying to load_clean_X....")

	with open('./Data/CleanX.pkl', 'rb') as f:
		X = pickle.load(f)

	return X
