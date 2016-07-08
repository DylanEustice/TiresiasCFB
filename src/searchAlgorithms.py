import numpy as np


def evolutionary_search(nPop, iters, kill_rate, evolve_rate, obj_fun, params, *args, **kwargs):
	"""
	"""
	# Initialize population
	init_pop = np.array(params).reshape(1, len(params))
	pop = evolve_pop(init_pop, evolve_rate, pop_inc=nPop-1)
	# Start iterating
	for i in range(iters):
		ixFitness, fitness = oracle(obj_fun, pop, *args, **kwargs)


def evolve_pop(pop, evolve_rate, pop_inc=1):
	"""
	"""
	rand_mat = np.random.random([int(pop_inc*pop.shape[0]), pop.shape[1]])
	pop_tile = np.tile(pop, (pop_inc, 1))
	offspring = pop_tile +pop_tile * evolve_rate * 2 * (rand_mat - 0.5)
	return np.vstack([pop, offspring])


def oracle(obj_fun, params, *args, **kwargs):
	fitness = []
	for row in range(params.shape[0]):
		fitness.append(obj_fun(params[row,:], *args, **kwargs))
	return np.argsort(fitness), fitness