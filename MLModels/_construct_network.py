import os
import numpy as np

from multiprocessing import current_process
if current_process().name == "MainProcess":
	from tensorflow.keras.optimizers import Adam
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.models import load_model
	from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU
	from tensorflow.keras.layers import Flatten, Dropout
	from tensorflow.keras.regularizers import l1, l2
	from tensorflow.keras import backend as K
	from tensorflow.math import argmax
	import tensorflow as tf



def construct_network(embedding_weights, vocab_size,**params):
	lr = params.get("lr")
	pretrained_emb = params.get("pretrained_emb")
	emb_dimension = params.get("emb_dimension")
	maxlen = params.get("maxlen")
	dropout = params.get("dropout")
	rec_dropout = params.get("rec_dropout")
	LSTM_cells = params.get("LSTM_cells")
	should_warm_up = params.get("should_warm_up")
	embedding_type = params.get("embedding_type")

	if should_warm_up:
		direc = f"./MLModels/TheModel-{embedding_type}.h5"
		print ("\n\n-----Model is loaded-----\n\n")
		if os.path.exists(direc):
			model = load_model(direc)
			return model


	print('constructing the deep neural network network...')

	if pretrained_emb:
		model = Sequential()

		model.add(Embedding(vocab_size, emb_dimension, input_length = maxlen,
		                weights = [embedding_weights], trainable = False))

		model.add(GRU(256, dropout = 0.5,
							recurrent_dropout = 0.25,
							return_sequences = True, name="gru_1"))
		model.add(GRU(128, dropout = 0.5,
							recurrent_dropout = 0.25,
							return_sequences = False,
							name="gru_2"))

		model.add(Dense(64, activation = 'relu', name="dense_1"))
		model.add(Dropout(0.5))

		model.add(Dense(32, activation = 'relu', name="dense_2"))
		model.add(Dropout(0.5))

		model.add(Dense(16, activation = 'relu', name="dense_3"))
		model.add(Dropout(0.5))

		model.add(Flatten())

		
		model.add(Dense(2, activation = 'softmax'))
		# model.summary()

	else:
		model = Sequential()

		model.add(Embedding(vocab_size, emb_dimension, input_length = maxlen))
		# model.add(LSTM(LSTM_cells[0], dropout = dropout, recurrent_dropout = rec_dropout, return_sequences = True))
		# model.add(LSTM(LSTM_cells[1], dropout = dropout, recurrent_dropout = rec_dropout, return_sequences = True))
		# model.add(LSTM(LSTM_cells[2], dropout = dropout, recurrent_dropout = rec_dropout, return_sequences = False))
		model.add(Dense(16, activation = 'relu'))
		model.add(Dense(1, activation = 'sigmoid'))
		# model.summary()

	opt = Adam(learning_rate = lr)

	model.compile(optimizer = opt,
					loss = 'binary_crossentropy',
					metrics = ['accuracy'])

	return model


def recall_m(y_true, y_pred):
	y_true = tf.gather(y_true, 1, axis = 1)
	y_pred = tf.gather(y_pred, 1, axis = 1)

	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision_m(y_true, y_pred):
	y_true = tf.gather(y_true, 1, axis = 1)
	y_pred = tf.gather(y_pred, 1, axis = 1)

	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1_m(y_true, y_pred):
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))