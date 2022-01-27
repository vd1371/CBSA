import pandas as pd

from ._get_polarity_of_contents_for_word_process import _get_polarity_of_contents_for_word_process
from utils import ParallelProcess

def _get_polarity_of_contents_for_word(df, ls_words, **params):

	n_cores = params.get("n_cores")

	df = ParallelProcess(df,
						_get_polarity_of_contents_for_word_process,
						ls_words,
						n_cores = n_cores)

	return df["sentiments"].values