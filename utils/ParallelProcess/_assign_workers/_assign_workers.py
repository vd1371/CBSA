import pandas as pd
import numpy as np
from multiprocessing import Queue, Process

from ._get_len_sections import _get_len_sections
from ._split_samples_for_processes import _split_samples_for_processes
from ._func_for_process import _func_for_process

def assign_workers(inp, func, *args, n_cores = None):

	n_cores = n_cores if n_cores != None else mp.cpu_count() - 2
	len_sections, n_cores = _get_len_sections(inp, n_cores)

	pool = []
	results_queue = Queue()

	print ("--- Assigning workers...")
	for process_number in range (n_cores):

		samples = _split_samples_for_processes(inp,
												process_number,
												len_sections)
		
		worker = Process(target = _func_for_process,
						args = (samples,
								func,
								process_number,
								results_queue,
								*args,))
		pool.append(worker)

	return pool, results_queue

