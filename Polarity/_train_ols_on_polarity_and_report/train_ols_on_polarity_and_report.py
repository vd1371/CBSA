import os
import pandas as pd

from .train_ols import train_ols

from DataLoader import load_polarity_file
from DataPreprocessing import get_hsi_for_polarity
from DataPreprocessing import get_jab_for_polarity
from DataPreprocessing import groupby_for_polarity

from utils import Logger

def train_ols_on_polarity_and_report(**params):

	report_dir = f"./reports/OLS"
	if not os.path.exists(report_dir):
		os.mkdir(report_dir)

	logger = Logger(address = report_dir + "/Log.log",
					mode = 'w')

	X = load_polarity_file()
	X = groupby_for_polarity(X)

	hsi = get_hsi_for_polarity(**params)
	jab = get_jab_for_polarity(**params)

	for df_name, output in {"hsi": hsi}.items():
							# "jab": jab}.items():
		for col in output.columns:
			print (f"Columns {col} of {df_name} is about to be analyzed")
			Y = output[col]
			df = pd.concat([X, Y], axis = 1, join = 'inner')
			X, Y = df.iloc[:, :-1], df.iloc[:, -1]

			train_ols(X, Y, logger, col, **params)