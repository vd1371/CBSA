from collections import Counter

def count_num_words(X, **params):
    count = Counter()

    for clause in X:
        for word in clause:
            count[word] += 1

    return count