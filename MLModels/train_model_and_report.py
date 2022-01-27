import pandas as pd


from DataPreprocessing import make_eng_col
from DataPreprocessing import segment_Y
from DataLoader import load_XYs

from embedding import load_network_info

from ._construct_network import construct_network
from ._train_model import train_model
from ._evaluate_classification import evaluate_classification
from ._generate_XY_batches import _generate_XY_batches
from ._get_callbacks import get_callbacks


def train_model_and_report(**params):

	index_dict, vocab_size, embedding_weights = \
			load_network_info(**params)

	embedding_type = params.get("embedding_type")
	model = construct_network(embedding_weights, vocab_size, **params)
	callback_list = get_callbacks(**params)

	for X_train, Y_train, X_test, Y_test in _generate_XY_batches(**params):

		# model = train_model(model, X_train, Y_train, callback_list, **params)
		pass
		
	evaluate_classification(X_test, Y_test,
							label = embedding_type,
							**params)