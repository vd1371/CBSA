from sklearn.model_selection import train_test_split

def split_val_test(X_test, Y_test, **params):
	X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.5,
													stratify = Y_test)

	return X_val, X_test, Y_val, Y_test
