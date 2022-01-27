

def join_processes(pool):

	print ("--- Workers tying to join...")
	for worker in pool:
		worker.join()