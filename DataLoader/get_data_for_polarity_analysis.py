import pandas as pd

from .get_data import get_data
from ._drop_content_null import _drop_content_null
from ._drop_non_chinese import _drop_non_chinese

def get_data_for_polarity_analysis(**params):

	n_samples = params.get("n_samples")

	direc = "./Data/analytics_challenge_dataset_ex210911.csv"
	small_direc = "./Data/analytics_challenge_dataset_ex210911-Small.csv"
	df = pd.read_csv(direc if n_samples == None else small_direc, index_col = 0)

	df = _drop_content_null(df)
	df = _drop_non_chinese(df, **params)
	df = df[~df['author_type'].isnull()]

	df['pubdate'] = pd.to_datetime(df['pubdate'])
	df['date'] = df['pubdate'].dt.date
	df = df.loc[:, ['date', 'content', 'author_type']]

	return df

