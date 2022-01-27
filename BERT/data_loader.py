from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def data_loader(X_train_seq, X_train_mask, Y_train,
			    X_val_seq, X_val_mask, Y_val,
			    **params):

	batch_size = params.get("batch_size")

	# wrap tensors
	X_train = TensorDataset(X_train_seq, X_train_mask, Y_train)

	# sampler for sampling the data during training
	train_sampler = RandomSampler(X_train)

	# dataLoader for train set
	train_dataloader = DataLoader(X_train,
								  sampler = train_sampler,
								  batch_size = batch_size)

	# wrap tensors
	X_val = TensorDataset(X_val_seq, X_val_mask, Y_val)

	# sampler for sampling the data during training
	val_sampler = SequentialSampler(X_val)

	# dataLoader for validation set
	val_dataloader = DataLoader(X_val,
								sampler = val_sampler,
								batch_size = batch_size)

	return train_dataloader, val_dataloader