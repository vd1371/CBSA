import re
import pandas as pd


def load_polarity(**params):
	data_directory = params.get("data_directory")
	polarity_dir = params.get("polarity_dir")

	df = pd.read_csv(data_directory)

	df['pubdate'] = pd.to_datetime(df['pubdate'])

	for idx, dateti in enumerate(df['pubdate']):
		df.loc[idx,'date'] = dateti.date()

	for idx, dateti in enumerate(df['pubdate']):
		df.loc[idx,'time'] = dateti.time()
	
	df = df.loc[:, ['date','time']]


	df_polarity = pd.read_csv(polarity_dir)

	holder = []

	for val in df_polarity['label']:
		int_val = int(re.search(r'\d+', val).group())
		holder.append(int_val)

	df_polarity = pd.DataFrame({
		'label' : holder,
		'score' : df_polarity['score']
		})

	df = pd.concat([df, df_polarity], axis = 1)

	return df
