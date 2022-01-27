
def _func_for_process(samples, func, process_number, results_queue, *args):

	results = func(samples, *args)
	results_queue.put((process_number, results))