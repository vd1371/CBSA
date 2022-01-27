import jieba

from .._replace_bad_chars import _replace_bad_chars

def _tokenize_samples(X,
					batch_number = None,
					q_out = None):

	tokenized = []

	for idx, sentence in enumerate(X):

		# sentence = _replace_bad_chars(sentence, replacements)
		tokenized_sentence = jieba.lcut(str(sentence), cut_all=True)
		tokenized.append(tokenized_sentence)

	if q_out == None:
		return tokenized
	else:
		q_out.put((batch_number, tokenized))

