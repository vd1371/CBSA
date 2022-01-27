# import matplotlib.pyplot as plt

# def plot_results(X, Y, results, **params):
# 	pred_ols = res.get_prediction()
# 	iv_l = pred_ols.summary_frame()["obs_ci_lower"]
# 	iv_u = pred_ols.summary_frame()["obs_ci_upper"]
# 	fig, ax = plt.subplots(figsize=(8, 6))

# 	ax.plot(X, Y, "o", label="data")
# 	ax.plot(X, y_true, "b-", label="True")
# 	ax.plot(X, res.fittedvalues, "r--.", label="OLS")
# 	ax.plot(X, iv_u, "r--")
# 	ax.plot(X, iv_l, "r--")
# 	ax.legend(loc="best")