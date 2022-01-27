import pandas as pd

def _drop_content_null(df, **params):

	print("dropping null values from content column")

	df = df[df['content'].notna()]

	return df

