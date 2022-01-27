

def train_bert_and_report():

	#option3: BERT
	# df = get_data(**settings)
	# df = make_eng_col(df, **settings)
	# X, Y = split_data(df, **settings)
	# Y = segment_Y(Y, **settings)
	# X = clean_text(X, **settings)
	# # X_train = remove_spaces(X_train)
	# X_train, X_test, Y_train, Y_test = train_test(X, Y, **settings)
	# X_val, X_test, Y_val, Y_test = split_val_test(X_test, Y_test, **settings)
	# X_train = tokenize_bert(X_train, **settings)
	# X_val = tokenize_bert(X_val, **settings)
	# X_test = tokenize_bert(X_test, **settings)
	# X_train_seq, X_train_mask, Y_train = sequence_to_tensor(X_train, Y_train, **settings)
	# X_val_seq, X_val_mask, Y_val = sequence_to_tensor(X_val, Y_val, **settings)
	# X_test_seq, X_test_mask, Y_test = sequence_to_tensor(X_test, Y_test, **settings)
	# train_dataloader, val_dataloader = data_loader(X_train_seq, X_train_mask, Y_train,
	# 		    								   X_val_seq, X_val_mask, Y_val, **settings)
	# bert = initialize_bert(X_train_seq, X_train_mask, **settings)
	# model, optimizer, weights, cross_entropy, device = fine_tune(bert, Y_train, **settings)
	# start_training(model, train_dataloader, val_dataloader, cross_entropy, optimizer, **settings)
	# predict_test(model, Y_test, **params)