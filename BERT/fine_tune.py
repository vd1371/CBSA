from .FineTune import *

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import AdamW

def fine_tune(bert, Y_train, **params):
	lr_bert = params.get("lr_bert")

	model = FineTune(bert)

	#push the model to GPU
	device = torch.device("cuda")
	model = model.to(device)

	optimizer = AdamW(model.parameters(), lr = lr_bert)

	class_wts = compute_class_weight('balanced', classes = np.unique(Y_train), y = Y_train.cpu().detach().numpy())

	# convert class weights to tensor
	weights= torch.tensor(class_wts, dtype = torch.float)
	weights = weights.to(device)
	
	cross_entropy  = nn.NLLLoss(weight = weights)

	return model, optimizer, weights, cross_entropy, device
	

