import numpy as np

def segment_Y(Y, **params):
	Y_segments = params.get("Y_segments")
	Y_quantile = params.get("Y_quantile")

	print("segmenting Y")

	Y = Y.values.reshape(-1)
	Y_quantile = np.quantile(Y, Y_quantile, axis = 0)

	bigger_mask = Y > Y_quantile
	smaller_mask = Y <= Y_quantile

	Y[bigger_mask] = 1
	Y[smaller_mask] = 0
	
	return Y