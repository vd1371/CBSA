import pandas as pd

from ._drop_content_null import _drop_content_null
from ._drop_non_chinese import _drop_non_chinese

def clean_data(**params):

	print ("Trying to clean_data")

	eng_cols = params.get("eng_cols")
	n_samples = params.get("n_samples")

	direc = "./Data/analytics_challenge_dataset_ex210911.csv"
	small_direc = "./Data/analytics_challenge_dataset_ex210911-Small.csv"
	df = pd.read_csv(direc if n_samples == None else small_direc)

	if not n_samples == None:
		df = df.iloc[:n_samples, :]

	df = df[(df['pubname'] == "Facebook - 群組 或專頁") | (df['pubname'] == "Facebook香港")]
	df = df[df['fans_count'] > 0]
	df = df[df['author_type'] == "媒體"]

	for col in eng_cols:
		df[col] = df[col].fillna(value = 0)
		
	df.drop(columns = ["docid", "author*", "pubname", "region"],
			inplace=True)

	df = _drop_content_null(df)
	df = _drop_non_chinese(df, **params)

	df.to_csv("./Data/analytics_challenge_dataset_ex210911-Clean.csv",
				encoding='utf-8-sig')