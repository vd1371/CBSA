import os
from multiprocessing import current_process
if current_process().name == "MainProcess":
	from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
	from ._PlotLosses import PlotLosses


def get_callbacks(**params):

	embedding_type = params.get("embedding_type")

	callback_list = []
	early_stopping = EarlyStopping(monitor='loss',
									min_delta = 0.01,
									patience=10,
									verbose=1,
									mode='auto')
	callback_list.append(early_stopping)

	plot_losses = PlotLosses()
	callback_list.append(plot_losses)

	checkpoint = ModelCheckpoint(
		f'./MLModels/TheModel-{embedding_type}-BestModel.h5',
		monitor='val_loss',
		verbose=1,
		save_best_only=True,
		mode='auto')
	callback_list.append(checkpoint)

	return callback_list