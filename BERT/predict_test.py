import numpy as np
from sklearn.metrics import classification_report
import torch

def predict_test(model, Y_test, **params):
	device = torch.device("cuda")

	path = 'saved_weights.pt'
	model.load_state_dict(torch.load(path))

	with torch.no_grad():
		preds = model(test_seq.to(device), test_mask.to(device))
		preds = preds.detach().cpu().numpy()

	# model's performance
	preds = np.argmax(preds, axis = 1)
	print(classification_report(Y_test, preds))