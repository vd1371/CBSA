import numpy as np
import torch
import gc
import time

def train_bert_model(model, train_dataloader, cross_entropy, optimizer, **params):

	device = torch.device("cuda")
  
	model.train()

	total_loss, total_accuracy = 0, 0

	# empty list to save model predictions
	total_preds = []

	start = time.time()
	# iterate over batches
	for step, batch in enumerate(train_dataloader):

		# progress update after every 50 batches.
		if step % 50 == 0 and not step == 0:
		  print(f'  Batch {step}  of  {len(train_dataloader)} in {time.time()-start:.2f}')
		  start = time.time()

		# push the batch to gpu
		batch = [r.to(device) for r in batch]

		sent_id, mask, labels = batch

		# clear previously calculated gradients 
		model.zero_grad()

		# get model predictions for the current batch
		# preds = model(sent_id, mask, labels)
		preds = model(sent_id, mask)

		# preds = torch.flatten((preds > 0.5).int())
		# preds = (preds > 0.5).float()

		labels = labels.unsqueeze(1).float()

		# compute the loss between actual and predicted values
		loss = cross_entropy(preds, labels)

		# add on to the total loss
		total_loss = total_loss + loss.item()

		# backward pass to calculate the gradients
		loss.backward()

		# clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

		# update parameters
		optimizer.step()

		# model predictions are stored on GPU. So, push it to CPU
		preds = preds.detach().cpu().numpy()

		# append the model predictions
		total_preds.append(preds)

		torch.cuda.empty_cache()
		gc.collect()

	# compute the training loss of the epoch
	avg_loss = total_loss / len(train_dataloader)

	# predictions are in the form of (no. of batches, size of batch, no. of classes).
	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds  = np.concatenate(total_preds, axis = 0)

	#returns the loss and predictions
	return avg_loss, total_preds        
