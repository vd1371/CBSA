import pandas as pd
import numpy as np

def make_eng_col(Y, **params):
	eng_cols = params.get("eng_cols")

	print('adding engagement column')

	Y['engagement'] = Y[eng_cols].sum(axis = 1)/np.log(Y['fans_count'])
	Y.drop(eng_cols + ['fans_count'], axis = 1, inplace = True)

	

	return Y