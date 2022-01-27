import os
import pandas as pd

def load_polarity_file():

	df = None
	if os.path.exists("./Data/Polarities.csv"):
		df = pd.read_csv("./Data/Polarities.csv", index_col = 0)

	return df