import numpy as np
from sklearn.model_selection import train_test_split


def train_test_split_(X, Y, **params):
	split_size = params.get("split_size")
	random_state = params.get("random_state")

	print("splitting data to train and test...")

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split_size,
		random_state = random_state, shuffle = True, stratify = Y)

	Y_train, Y_test = np.array(Y_train), np.array(Y_test)
	
	return X_train, X_test, Y_train, Y_test