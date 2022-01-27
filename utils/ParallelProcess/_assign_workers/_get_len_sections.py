

def _get_len_sections(inp, n_cores):

	L = len(inp)
	l_sections = int(L/n_cores) + 1

	if (n_cores - 1) * l_sections > L-2:
		n_cores = int(L/l_sections)
		return l_sections, n_cores
	else:
		return l_sections, n_cores