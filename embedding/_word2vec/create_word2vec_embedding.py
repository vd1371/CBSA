import pickle

from ._train_word2vec import train_word2vec
from ._emb_matrix_wv import _emb_matrix_wv

from .._save_network_info import _save_network_info

from DataLoader import load_tokenized_X

def create_word2vec_embedding(**params):

	print ("Trying to create_word2vec_embedding...")

	X = load_tokenized_X(**params)

	index_dict, word_vectors = train_word2vec(X, **params)
	vocab_size, embedding_weights = _emb_matrix_wv(index_dict,
												word_vectors,
												**params)

	_save_network_info(index_dict, vocab_size, embedding_weights, 'word2vec')

	



