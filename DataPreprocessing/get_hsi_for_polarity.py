import pandas as pd
from DataLoader import get_hsi

def get_hsi_for_polarity(**params):

	Y = pd.DataFrame()

	hsi = get_hsi(**params)

	today_close = hsi['Close']
	tomorrow_close = hsi['Close'].shift(-1)

	Y['Tom-Tod'] = tomorrow_close - today_close
	Y['(Tom-Tod)/Tod'] = (tomorrow_close - today_close)/today_close
	diff_perc = (tomorrow_close - today_close)/today_close

	pos_mask = diff_perc > 0.02
	neg_mask = diff_perc < -0.02
	neu_mask = ((diff_perc < 0.02) & (diff_perc > -0.02))
	diff_perc[pos_mask] = 1
	diff_perc[neg_mask] = -1
	diff_perc[neu_mask] = 0
	Y['Diff-Discrete'] = diff_perc
	Y.dropna(inplace = True)
	Y.index = pd.to_datetime(Y.index)

	return Y