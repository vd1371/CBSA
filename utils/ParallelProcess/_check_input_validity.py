import numpy as np
import pandas as pd

def _check_input_validity(inp):

	assert isinstance(inp, (pd.DataFrame, pd.Series, list, np.ndarray)),\
			"The input value must be pandas DataFrame, list or numpy array"