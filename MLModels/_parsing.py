import gc

from utils import ParallelProcess

def parsing(X, w2indx, **params):

    print("parsing the tokenized sentences...")
    n_cores = params.get("n_cores")
    results = ParallelProcess(X,
                                _parsing_for_parallel,
                                w2indx,
                                n_cores = n_cores)

    return results

def _parsing_for_parallel(X, w2indx):

    holder = []
    for sentence in X:
        indexed_sentence = []

        for token in sentence:
            try:
                indexed_sentence.append(w2indx[token])
            except:
                indexed_sentence.append(0)

        holder.append(indexed_sentence)

    return holder