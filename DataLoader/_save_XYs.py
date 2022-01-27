import pickle

def save_XYs(X_train, Y_train, X_test, Y_test, **params):

	print ("Trying to save_XYs...")

	embedding_type = params.get("embedding_type")

	base_dir = './Data/'
	with open(base_dir + f"{embedding_type}_X_train.pkl", 'wb') as f:
		pickle.dump(X_train, f)

	with open(base_dir + f"{embedding_type}_Y_train.pkl", 'wb') as f:
		pickle.dump(Y_train, f)

	with open(base_dir + f"{embedding_type}_X_test.pkl", 'wb') as f:
		pickle.dump(X_test, f)

	with open(base_dir + f"{embedding_type}_Y_test.pkl", 'wb') as f:
		pickle.dump(Y_test, f)



