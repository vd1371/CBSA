import json
import pickle

def load_unique_words(**params):

	print ("Trying to load_unique_words")
	# with open("./Data/unique.json", 'r', encoding = 'utf-8-sig') as f:
	# 	unique_words = json.load(f)

	with open("./Data/unique.pkl", 'rb') as f:
		unique_words = pickle.load(f)

	return unique_words