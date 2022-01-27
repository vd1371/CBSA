import gensim

from ._emb_dict_wv import _emb_dict_wv

def train_word2vec(X, **params):
	emb_dimension = params.get("emb_dimension")
	min_word_count_wv = params.get("min_word_count_wv")
	window_size = params.get("window_size") 
	skipgram = params.get("skipgram")
	wv_epochs = params.get("wv_epochs")
	maxlen = params.get("maxlen")

	print("training word2vec model")

	wv_model = gensim.models.word2vec.Word2Vec(sentences = X,
								   			vector_size = emb_dimension,
								   			window = window_size,
								   			min_count = min_word_count_wv,
								   			sg = skipgram,
								   			epochs = wv_epochs,
								   			)

	wv_model.build_vocab(X)

	wv_model.train(X, total_examples=wv_model.corpus_count,
		epochs = wv_epochs,)
	
	wv_model.save("./embedding/_word2vec/w2v_model.pkl")

	index_dict, word_vectors = _emb_dict_wv(X = X, wv_model = wv_model)

	return index_dict, word_vectors