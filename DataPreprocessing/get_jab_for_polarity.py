import pandas as pd

def get_jab_for_polarity(**params):

	df = pd.read_csv("./Data/Jab.csv")

	df.loc[:, "Jabs"] = df['Sinovac 1st dose'] + df['BioNTech 1st dose']
	df.drop(columns = ["Sinovac 1st dose", "Sinovac 2nd dose",
						"Sinovac 3rd dose", "BioNTech 1st dose",
						"BioNTech 2nd dose","BioNTech 3rd dose"],
			inplace = True)

	today_df = df.groupby(["Age Group", "Date"]).sum().unstack(level=0)
	today_df.columns = today_df.columns.to_flat_index()

	tomorrow_df = today_df.shift(-1)


	jab_df = (tomorrow_df-today_df)/ (today_df + 0.00000001)
	jab_df.dropna(axis = 0, inplace = True)


	jab_df.index = pd.to_datetime(jab_df.index)

	return jab_df

	