from ._train_bert_model import train_bert_model
from ._evaluate import evaluate

import torch

def start_training(model, train_dataloader, val_dataloader, cross_entropy, optimizer, **params):
	bert_epochs = params.get("bert_epochs")

	# set initial loss to infinite
	infinite = float('inf')

	# empty lists to store training and validation loss of each epoch
	train_losses = []
	valid_losses = []

	#for each epoch
	for epoch in range(bert_epochs):
	     
	    print('\n Epoch {:} / {:}'.format(epoch + 1, bert_epochs))
	    
	    #train model
	    train_loss, _ = train_bert_model(model, train_dataloader, cross_entropy, optimizer, **params)
	    
	    #evaluate model
	    valid_loss, _ = evaluate(model, val_dataloader, cross_entropy, **params)
	    
	    #save the best model
	    if valid_loss < infinite:
	        infinite = valid_loss
	        torch.save(model.state_dict(), 'saved_weights.pt')
	    
	    # append training and validation loss
	    train_losses.append(train_loss)
	    valid_losses.append(valid_loss)
	    
	    print(f'\nTraining Loss: {train_loss:.3f}')
	    print(f'Validation Loss: {valid_loss:.3f}')

