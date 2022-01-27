import numpy as np

def contents_to_str(X, **params):

	print("converting tokenized contents to one string for each...")

	holder = []
	
	if len(X) < 512:
		for content in X:
			list_to_str = ' '.join([str(elem) for elem in content])
			holder.append(list_to_str)

	else:
		while len(X) < 512:
			X.pop(np.random.randint(low = 0, high = 512))

		for content in X:
			list_to_str = ' '.join([str(elem) for elem in content])
			holder.append(list_to_str)

	return holder