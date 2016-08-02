import numpy as np

def evolutionary_search(nPop, iters, kill_rate, evolve_rng, obj_fun, params, *args, **kwargs):
	"""
	"""
	# Initialize population
	init_pop = np.array(params).reshape(1, len(params))
	pop = evolve_pop(init_pop, evolve_rng, pop_inc=nPop)
	if int(nPop * kill_rate) != nPop * kill_rate:
		tmp_nPop = nPop
		nPop = int(int(nPop * kill_rate) / kill_rate)
		print "Reducing population size from {} to {}".format(tmp_nPop, nPop)
	nSurvive = int(nPop * kill_rate)
	nPop_inc = nPop / nSurvive
	best_fit = []
	# Start iterating
	for i in range(iters):
		if i % 1 == 0:
			print "iter {} of {}".format(i, iters)
		fitness = oracle(obj_fun, pop, *args, **kwargs)
		ixFit = np.argsort(fitness)
		tmp_pop = np.vstack([pop[ix,:] for ix in ixFit[:nSurvive]])
		pop = evolve_pop(tmp_pop, evolve_rng, pop_inc=nPop_inc)
		best_fit.append(fitness[ixFit[0]])
	return pop, best_fit


def evolve_pop(pop, evolve_rng, pop_inc=1):
	"""
	"""
	rand_mat = np.random.random([int((pop_inc-1)*pop.shape[0]), pop.shape[1]])
	pop_tile = np.tile(pop, (pop_inc-1, 1))
	offspring = pop_tile + pop_tile * evolve_rng * 2 * (rand_mat - 0.5)
	return np.vstack([pop, offspring])


def oracle(obj_fun, params, *args, **kwargs):
	"""
	"""
	fitness = []
	for row in range(params.shape[0]):
		fitness.append(obj_fun(params[row,:], *args, **kwargs))
	return fitness