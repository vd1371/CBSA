from ._tokenize_samples import _tokenize_samples
from ._tokenize_samples_parallel import _tokenize_samples_parallel

def tokenize(X, **params):

	n_cores = params.get("n_cores")
	print("Trying to tokenize ...")
	
	tokenized = []

	if n_cores == 1:
		tokenized = _tokenize_samples(X)

	else:
		tokenized = _tokenize_samples_parallel(X, **params)

	return tokenized