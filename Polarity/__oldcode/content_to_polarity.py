from transformers import AutoModelForSequenceClassification, pipeline


def content_to_polarity(X, **params):
	sentiment_checkpoint = params.get("sentiment_checkpoint")
	
	classifier = pipeline('sentiment-analysis', model = sentiment_checkpoint)

	X_label = []
	X_score = []
	counter = 0

	for idx, sentence in enumerate(X):
		
		try:
			if idx % 1000 == 0:
				print(f'got polarity of {idx} contents')

			output = classifier(sentence)
			X_label.append(output[0]['label'])
			X_score.append(output[0]['score'])
		
		except Exception as e:
			print(sentence)
			print('**************************************',type(sentence))
			print(e)
			raise ValueError
			counter += 1
			pass

	print(f"couldn't got polarity of {counter} sentences, because of their length")

	return X_label, X_score
