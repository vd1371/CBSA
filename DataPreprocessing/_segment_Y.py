import numpy as np

def segment_Y(Y, **params):
	Y_segments = params.get("Y_segments")
	Y_quantile = params.get("Y_quantile")

	print("segmenting Y")

	Y = Y.values.reshape(-1)
	Y_quantile = np.quantile(Y, Y_quantile, axis = 0)

	bigger_mask = (Y > Y_quantile).copy()
	smaller_mask = (Y <= Y_quantile).copy()

	Y[bigger_mask] = 1
	Y[smaller_mask] = 0

	Y = Y.astype(int)

	return Y