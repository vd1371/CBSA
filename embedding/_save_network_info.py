import pickle

def _save_network_info(index_dict,
						vocab_size,
						embedding_weights,
						embedding_type):

	if not embedding_type in ['word2vec', 'fasttext']:
		raise ValueError ("embedding_type MUST be 'word2vec' or 'fasttext'")


	base_dir = f"./embedding/_{embedding_type}/"
	with open(base_dir + 'index_dict.pkl', 'wb') as handle:
		pickle.dump(index_dict, handle)

	with open(base_dir + 'vocab_size.pkl', 'wb') as handle:
		pickle.dump(vocab_size, handle)

	with open(base_dir + 'embedding_weights.pkl', 'wb') as handle:
		pickle.dump(embedding_weights, handle)