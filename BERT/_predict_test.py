import numpy as np
from sklearn.metrics import classification_report
import torch

from MLModels import evaluate_classification

def predict_test(model, X_test, Y_test, **params):
	device = torch.device("cuda")

	path = 'saved_weights.pt'
	model.load_state_dict(torch.load(path))

	with torch.no_grad():
		preds = model(test_seq.to(device), test_mask.to(device))
		preds = preds.detach().cpu().numpy()

	evaluate_classification(X_test, Y_test, "Bert", **params)
