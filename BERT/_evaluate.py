import numpy as np
import torch

def evaluate(model, val_dataloader, cross_entropy, **params):
	device = torch.device("cuda")

	print("\nEvaluating...")

	# deactivate dropout layers
	model.eval()

	total_loss, total_accuracy = 0, 0

	# empty list to save the model predictions
	total_preds = []

	# iterate over batches
	for step,batch in enumerate(val_dataloader):

		# Progress update every 50 batches.
		# if step % 50 == 0 and not step == 0:
		  
			# Calculate elapsed time in minutes.
			# elapsed = format_time(time.time() - t0)
			    
			# Report progress.
			# print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

		# push the batch to gpu
		batch = [t.to(device) for t in batch]

		sent_id, mask, labels = batch

		# deactivate autograd
		with torch.no_grad():
		  
			# model predictions
			preds = model(sent_id, mask)
			# compute the validation loss between actual and predicted values

			labels = labels.unsqueeze(1).float()
			
			loss = cross_entropy(preds,labels)

			total_loss = total_loss + loss.item()

			preds = preds.detach().cpu().numpy()

			total_preds.append(preds)

	# compute the validation loss of the epoch
	avg_loss = total_loss / len(val_dataloader) 

	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds  = np.concatenate(total_preds, axis = 0)

	return avg_loss, total_preds