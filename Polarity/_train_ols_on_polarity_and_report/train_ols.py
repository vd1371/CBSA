import statsmodels.api as sm
import time

def train_ols(X, Y, logger, col, **params):

	X = sm.add_constant(X)

	model = sm.OLS(Y, X)
	# results = model.fit_regularized(alpha=1., L1_wt=0.5)
	results = model.fit()

	logger.info(str(col))
	logger.info((str(results.summary())))