

def start_processes(pool_of_workers):

	for worker in pool_of_workers:
		worker.start()
	print ("--- Workers started")