
def save_polarity_to_file(df):
	print ("Trying to savel polarities file")
	df.drop(columns = ['content'], inplace = True, errors = 'ignore')

	df.to_csv("./Data/Polarities.csv", encoding = 'utf-8-sig')