
def train_model(model, X_train, Y_train, callback_list, **params):
	train_epochs = params.get("train_epochs")
	batch_size = params.get("batch_size")
	val_split = params.get("val_split")
	model_verbose = params.get("model_verbose")
	embedding_type = params.get("embedding_type")

	print("training neural network...")

	model.fit(X_train, Y_train,
				epochs = train_epochs,
                batch_size = batch_size,
                validation_split = val_split,
                verbose = model_verbose,
                shuffle=True,
                callbacks=callback_list)

	model.save(f"./MLModels/TheModel-{embedding_type}.h5")

	return model