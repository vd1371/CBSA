from transformers import AutoModel#, BertForSequenceClassification

def initialize_bert(sent_id, mask, **params):
	checkpoint = params.get("checkpoint")

	bert = AutoModel.from_pretrained(checkpoint)
	
	for param in bert.parameters():
		param.requires_grad = False

	return bert