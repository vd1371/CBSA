import pandas as pd

def groupby_for_polarity(X):

	# X.loc[X['author_type'] == "企業品牌", 'author_type']= "Bus" #"EnterpriseBrands"
	# X.loc[X['author_type'] == "商界名人", 'author_type'] = "Bus" #"BusinessCelebrity"
	# X.loc[X['author_type'] == "媒體", 'author_type'] = "Media"
	# X.loc[X['author_type'] == "政界人士", 'author_type'] = "Politicians"
	# X.loc[X['author_type'] == "機構及社群", 'author_type'] = "PPL" #"InstitutionsAndCommunities"
	# X.loc[X['author_type'] == "演藝明星", 'author_type'] = "PPL" #"EntertainmentStar"
	# X.loc[X['author_type'] == "網民", 'author_type'] = "PPL" #"Netizens"
	# X.loc[X['author_type'] == "網紅博客", 'author_type'] = "PPL" #"InfluencerBlog"

	X.loc[X['author_type'] == "企業品牌", 'author_type']= "EnterpriseBrands"
	X.loc[X['author_type'] == "商界名人", 'author_type'] = "BusinessCelebrity"
	X.loc[X['author_type'] == "媒體", 'author_type'] = "Media"
	X.loc[X['author_type'] == "政界人士", 'author_type'] = "Politicians"
	X.loc[X['author_type'] == "機構及社群", 'author_type'] = "InstitutionsAndCommunities"
	X.loc[X['author_type'] == "演藝明星", 'author_type'] = "EntertainmentStar"
	X.loc[X['author_type'] == "網民", 'author_type'] = "Netizens"
	X.loc[X['author_type'] == "網紅博客", 'author_type'] = "InfluencerBlog"

	X = X.groupby(['author_type', "date"]).mean().unstack(level=0)
	X.fillna(0.5, inplace = True)

	X.columns = X.columns.to_flat_index()
	X.index = pd.to_datetime(X.index)

	return X