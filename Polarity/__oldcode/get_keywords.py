import json

def get_keywords(X, **params):
	keywords_dir = params.get("keywords_dir")

	with open(keywords_dir, 'r', encoding = 'utf-8-sig') as f:
		keywords = json.load(f)

	sentences = []

	for sentence in X:
		sentences.append([token for token in sentence if token in keywords])

	return sentences
