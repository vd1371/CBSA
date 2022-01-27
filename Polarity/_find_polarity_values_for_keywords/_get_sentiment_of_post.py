from snownlp import SnowNLP

def _get_sentiment_of_post(post, ls_words):

	for word in ls_words:
		if word in post:
			text = SnowNLP(post)
			return text.sentiments
	return 0.5


	sentences = text.sentences

	sentiment = 0
	if len(sentences) == 0:
		return sentiment
	else:
		for sentence in sentences:
			s = SnowNLP(sentence)
			sentiment += s.sentiments

		print (text.sentiments, sentiment/len(sentences))

		return sentiment/len(sentences)