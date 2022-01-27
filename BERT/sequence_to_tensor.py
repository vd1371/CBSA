import torch

def sequence_to_tensor(X, Y, **params):
	X_seq = torch.tensor(X['input_ids'])
	X_mask = torch.tensor(X['attention_mask'])
	Y = torch.tensor(Y.tolist())

	return X_seq, X_mask, Y