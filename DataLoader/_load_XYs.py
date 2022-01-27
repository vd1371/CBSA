import pickle

from sklearn.utils import shuffle

def load_XYs(**params):

	print ("Trying to load_XYs...")

	embedding_type = params.get("embedding_type")

	base_dir = './Data/'
	with open(base_dir + f"{embedding_type}_X_train.pkl", 'rb') as f:
		X_train = pickle.load(f)

	with open(base_dir + f"{embedding_type}_Y_train.pkl", 'rb') as f:
		Y_train = pickle.load(f)

	with open(base_dir + f"{embedding_type}_X_test.pkl", 'rb') as f:
		X_test = pickle.load(f)

	with open(base_dir + f"{embedding_type}_Y_test.pkl", 'rb') as f:
		Y_test = pickle.load(f)

	X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

	return X_train, Y_train, X_test, Y_test