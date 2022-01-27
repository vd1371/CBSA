from sklearn.preprocessing import RobustScaler, StandardScaler

import numpy as np

def scaler(Y, **params):
	Y = np.array(Y)
	Y = np.reshape(Y, (-1, 1))

	_scaler = StandardScaler()

	Y = _scaler.fit_transform(Y)

	return Y