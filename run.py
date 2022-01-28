from DataLoader import *
from DataPreprocessing import *
from embedding import *
from MLModels import *
# from utils import *
# from BERT import *
# from Polarity import *

import os
import time

import pandas as pd


def run(**params):
	settings = {
	"n_cores": 10,
	"n_samples": None,
	'embedding_type': 'fasttext',

	"batch_size" : 256,
	"should_warm_up": False,
	"train_epochs" : 100,
	"lr": 0.001,

	"eng_cols" : ["comment_count", "like_count", "dislike_count",
					"love_count", "haha_count", "wow_count",
					"angry_count", "sad_count", "share_count",
					"view_count"],
	"Y_segments" : 2,
	"Y_quantile" : 0.8,
	"emb_dimension" : 300,
	"maxlen" : 100, # To be subjectively optimized
	'should_plot_length': False,
	"min_word_count_wv" : 2,
	"skipgram" : 0,
	"wv_epochs" : 5,
	"window_size" : 5,

	"split_size" : 0.2,
	"k_nbrs_overs" : 5,
	"random_state" : 42,
	"pretrained_emb" : True,
	"dropout" : 0.5,
	"rec_dropout" : 0.25,
	"LSTM_cells" : [16, 32, 16],
	"lr_bert": 0.00002,
	"bert_epochs" : 10,
	"val_split" : 0.2,
	"model_verbose" : 2,
	"checkpoint" : "hfl/chinese-bert-wwm-ext",
	# "sentiment_checkpoint" : "techthiyanes/chinese_sentiment",
	# "covid_checkpoint" : "yaoyinnan/bert-base-chinese-covid19",
	"bert_padding" : 64,

	}

	# One-time Run
	# clean_data(**settings)
	# load_original_file_and_save_as_tokenized(**settings)
	# find_unique_words_and_save_to_json(**settings)
	# create_word2vec_embedding(**settings)
	# create_fasttext_embedding(**settings)
	# parse_pad_and_save_x_train_test(**settings)
	
	# Each time
	train_model_and_report(**settings)
	# train_bert_and_report(**settings)

	#Polarity
	# df = get_data_for_polarity_analysis(**settings)
	# find_polarity_values_for_keywords(df, **settings)
	# train_ols_on_polarity_and_report(**settings)


if __name__ == '__main__':
	run()
	# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	print("Done")
