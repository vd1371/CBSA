from ._pkl_embedding import pkl_embedding
from ._emb_matrix_fasttext import emb_matrix_fasttext

from .._save_network_info import _save_network_info

from DataLoader import load_tokenized_X

def create_fasttext_embedding(**params):

	X = load_tokenized_X(**params)

	pkl_embedding(**params)
	index_dict, vocab_size, embedding_matrix = \
		emb_matrix_fasttext(X, **params)

	_save_network_info(index_dict, vocab_size, embedding_matrix, "fasttext")



