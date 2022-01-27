import pandas as pd

def get_hsi(**params):

	df = pd.read_csv("./Data/HSI.csv", index_col = 0)
	df.dropna(axis=0, inplace = True)

	return df