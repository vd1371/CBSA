from ._tokenize_samples import _tokenize_samples

from utils import ParallelProcess

def _tokenize_samples_parallel(X, **params):

	n_cores = params.get("n_cores")

	results = ParallelProcess(X,
								_tokenize_samples,
								n_cores = n_cores)

	return results

