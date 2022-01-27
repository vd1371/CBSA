import numpy as np
import matplotlib.pyplot as plt

def plot_length(X, **params):
	plot_start = params.get("plot_start", 0)
	plot_stop = params.get("plot_stop", 400)
	plot_step = params.get("plot_step", 4)

	totalNumWords = [len(sentence) for sentence in X]
	
	plt.hist(totalNumWords, 
			bins = np.arange(plot_start, plot_stop, plot_step))

	plt.show()