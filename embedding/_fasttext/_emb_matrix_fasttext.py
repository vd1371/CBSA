import os
import numpy as np

from multiprocessing import current_process
if current_process().name == "MainProcess":
    from tensorflow.keras.preprocessing.text import Tokenizer

from ._load_embedding import _load_embedding

from DataLoader import load_unique_words

def emb_matrix_fasttext(X, **params):
    emb_dimension = params.get("emb_dimension")

    vocab_size = len(load_unique_words()) + 1

    print('creating embedding matrix')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    index_dict = tokenizer.word_index
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, emb_dimension))

    word_embedding = _load_embedding(**params)

    for word, index in index_dict.items():
        #word is the key and i is the value of tokenizer.word_index.items() dictionary
        embedding_vector = word_embedding[emb_dimension].get(word)
        
        if index < vocab_size:
            if embedding_vector is not None:
                    #words not found in embedding index will be all-zeros
                    embedding_matrix[index] = embedding_vector

    return index_dict, vocab_size, embedding_matrix

