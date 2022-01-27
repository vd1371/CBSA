
def _replace_all(sentence, replacements):
    
    for i, j in replacements.items():
        sentence = str(sentence).replace(i, j)
    
    return sentence


def clean_text(X, **params):
	replacements = params.get("replacements")
		
	sentences = []

	for sentence in list(X):
		sentence = _replace_all(sentence, replacements)
		sentences.append(sentence)

	return sentences