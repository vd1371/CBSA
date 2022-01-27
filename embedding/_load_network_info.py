import pickle

def load_network_info(**params):

	embedding_type = params.get("embedding_type")

	if not embedding_type in ['word2vec', 'fasttext']:
		raise ValueError ("embedding_type MUST be 'word2vec' or 'fasttext'")

	base_dir = f"./embedding/_{embedding_type}/"

	with open(base_dir + 'index_dict.pkl', 'rb') as handle:
		index_dict = pickle.load(handle)

	with open(base_dir + 'vocab_size.pkl', 'rb') as handle:
		vocab_size = pickle.load(handle)

	with open(base_dir + 'embedding_weights.pkl', 'rb') as handle:
		embedding_weights = pickle.load(handle)

	return index_dict, vocab_size, embedding_weights