from sklearn.preprocessing import RobustScaler, StandardScaler

import numpy as np

def _scaler(Y,**params):
	Y = np.array(Y)
	Y = np.reshape(Y, (-1, 1))

	scaler = StandardScaler()

	Y = scaler.fit_transform(Y)

	return Y