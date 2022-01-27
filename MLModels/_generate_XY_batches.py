import gc
import numpy as np
from DataLoader import load_XYs

from multiprocessing import current_process
if current_process().name == "MainProcess":
	from tensorflow.keras.utils import to_categorical


def _generate_XY_batches(**params):

	n_big_batches = 1
	for i in range(n_big_batches):
		
		X_train, Y_train, X_test, Y_test = load_XYs(**params)

		L = len(X_train)
		l_sections = int(L/n_big_batches) + 1

		X_batch = X_train[i*l_sections: min((i+1)*l_sections, L)]
		Y_batch = Y_train[i*l_sections: min((i+1)*l_sections, L)]

		Y_batch = to_categorical(Y_batch)
		Y_test = to_categorical(Y_test)

		del X_train
		del Y_train
		gc.collect()

		yield X_batch, Y_batch, X_test, Y_test

