
def _remove_space_from_tokenized_sentence(_list,**params):

	removements = ['', ' ',]

	for obj in removements:

		while obj in _list:
			_list.remove(obj)

	return _list


def remove_space_from_all_samples(X, **params):
	sentences = []

	print("trying to remove_space_from_all_samples")

	for sentence in X:
		_remove_space_from_tokenized_sentence(sentence)
		sentences.append(sentence)

	return sentences