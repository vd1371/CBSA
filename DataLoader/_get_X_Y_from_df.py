def get_X_Y_from_df(df, **params):

	print("Getting X and Y from data")

	X = df['content']
	Y = df[params.get("eng_cols")]

	return X.copy(), Y.copy()