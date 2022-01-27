import pickle

def _load_embedding(**params):
	base_dir = "./embedding/_fasttext/"
	emb_dimension = params.get("emb_dimension")

	word_embedding = {300:{}, 200:{}, 100:{}}

	with open(base_dir + \
				f'fasttext_embedding_{emb_dimension}_pretrained.pkl',
				'rb') as f:
		word_embedding[emb_dimension] = pickle.load(f)
		
	return word_embedding[emb_dimension]