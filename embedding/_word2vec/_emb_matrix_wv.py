import numpy as np

def _emb_matrix_wv(index_dict, word_vectors, **params):
    emb_dimension = params.get('emb_dimension')

    print("creating embedding matrix for word2vec")
    
    vocab_size = len(index_dict) + 1

    embedding_weights = np.zeros((vocab_size, emb_dimension))

    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
        
    return vocab_size, embedding_weights