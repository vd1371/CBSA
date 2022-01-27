from imblearn.over_sampling import SMOTE

def oversample(X_train, Y_train, **params):
	k_nbrs_overs = params.get("k_nbrs_overs")
	
	smote = SMOTE(sampling_strategy = 'minority', k_neighbors = k_nbrs_overs)

	X_train, Y_train = smote.fit_resample(X_train, Y_train)

	return X_train, Y_train