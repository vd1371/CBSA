import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from utils import Logger

from multiprocessing import current_process
if current_process().name == "MainProcess":
	from tensorflow.keras.models import load_model

def evaluate_classification(x, y_true, label, **params):

	report_dir = f"./reports/{label}"
	if not os.path.exists(report_dir):
		os.mkdir(report_dir)
	logger = Logger(address = report_dir + "/Log.log")


	direc = f"./MLModels/TheModel-{label}-BestModel.h5"
	model = load_model(direc)


	y_pred = np.argmax(model.predict(x), axis = 1)
	y_true = np.argmax(y_true, axis = 1)

	logger.info(f"--------Classification Report for - {label}----------\n" + \
	               str(classification_report(y_true, y_pred))+"\n")
	logger.info(f"--------Confusion Matrix for - {label}----------\n" + \
	               str(confusion_matrix(y_true, y_pred))+"\n")
	logger.info(f'--------Accurcay for {label}----------\n' + \
	               str(round(accuracy_score(y_true, y_pred),4)))

	print (classification_report(y_true, y_pred))
	print (f'Accuracy score for - {label}',
			round(accuracy_score(y_true, y_pred),4))
	print ("------------------------------------------------")