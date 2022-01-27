import time

from ._get_sentiment_of_post import _get_sentiment_of_post

def _get_polarity_of_contents_for_word_process(df, ls_words):

	sentiments = []
	for post in df['content']:
		sentiments.append(_get_sentiment_of_post(post, ls_words))

	df['sentiments'] = sentiments

	return df



