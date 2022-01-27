import multiprocessing as mp

from ._assign_workers import assign_workers
from ._get_the_results import get_the_results
from ._start_processes import start_processes
from ._join_processes import join_processes
from ._check_input_validity import _check_input_validity
from ._merge_results import merge_results


def ParallelProcess(inp, func, *args, n_cores = None):
	'''
	This is a simplified multiprocessing function for lists of lists or a 
	datframe

	inp:
		lists of lists, 2d numpy array, or pandas
	func:
		the function that needs to be applied to the inp. The function does
		not return a different value but returns the modified version of the
		inp. In case you want something from the function, convert the inp to
		dataframe and add it as a column to it. The first argument of the func
		must be inp, otherwise unexpected behavior might happen.
	*args:
		the arguments that the function recieves for conducting th analysis
	n_cores:
		number of processes that will be run in parallel.
		Default: mp.cpu_count() - 2
	::returns:: the modified version of inp
	'''
	_check_input_validity(inp)
	print (f"Trying to analyze {func.__name__} in parallel")

	pool_of_workers, results_queue = \
			assign_workers(inp, func, *args, n_cores = n_cores)
	start_processes(pool_of_workers)
	results = get_the_results(pool_of_workers, results_queue)
	join_processes(pool_of_workers)
	results = merge_results(results)

	return results



	
