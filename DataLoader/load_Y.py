from .get_data import get_data
from ._get_X_Y_from_df import get_X_Y_from_df

def load_Y(**params):

	df = get_data(**params)
	X, Y = get_X_Y_from_df(df, **params)
	Y['fans_count'] = df.loc[:, 'fans_count']

	return Y

