from transformers import AutoTokenizer

def tokenize_bert(X, **params):
	checkpoint = params.get("checkpoint")
	bert_padding = params.get("bert_padding")

	tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
	
	tokenized = tokenizer.batch_encode_plus(X, 
											padding = 'max_length',
										  max_length = bert_padding,
										  truncation = True, 
										  return_token_type_ids = False)

	return tokenized

