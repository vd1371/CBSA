import pandas as pd


def polarity_to_df(X_label, X_score, **params):
	polarity_dir = params.get("polarity_dir")

	print('making a dataframe for polarities and scores and save it as a .csv file')

	df = pd.DataFrame({
			'label' : X_label,
			'score' : X_score
			}, columns = ['label', 'score'])

	df.to_csv(polarity_dir, index=False, header=True)
