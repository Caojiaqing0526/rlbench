import numpy as np 

from environment import Environment


class GridEnvironment(Environment):
	def __init__(self, *args, **kwargs):
		pass

	def _adjacent(self, node):
		"""The nodes connected by a single edge to `node`.""" 
		pass

	def _level(self, node, d):
		"""The nodes within a given distance `d` of `node`."""
		pass

	def _distance(self, u, v):
		pass

	def _bfs(self, node, max_depth):
		pass


def moore(r, dims):
	"""Moore neighborhood within distance `r` for `dims` dimensions.

	Makes use of the fact that Python treats `0` as false, `1` as true.
	"""

	def recurse(array, tmp, d):
		if d == dims - 1:
			for i in range(-r, r+1):
				if i or any(tmp):
					array.append(tuple(tmp + [i]))
		else:
			for i in range(-r, r+1):
				recurse(array, tmp + [i], d+1)
		return array

	return recurse([], [], 0) 


def check_moore(r, dims):
	array = moore(r, dims)
	# Check no illegal distances
	for i in array:
		for j in i:
			if abs(j) > r:
				print(i)
	# Check that all nodes enumerated
	basis = [i for i in range(-r, r)]
	from itertools import product
	
	alt = list(product(range(-r, r+1), repeat=dims))
	alt.remove(tuple([0 for i in range(dims)]))

	for i in alt:
		if i not in array:
			print(i)



def von_neumann(r, dims):
	"""Von Neumann neighborhood within distance `r` for `dims` dimensions."""


def check_von_neumann(r, dims):
	array = von_neumann(r, dims)
	for i in array:
		tmp = 0
		for j in i:
			tmp += abs(j)
		if tmp > r:
			print(tmp, i)
