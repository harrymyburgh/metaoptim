import copy
import numpy as np


class PSO:
    def __init__(self, problem, swarm_size, dim, max_iter, minimize=True):
        '''
        Initialize the PSO algorithm.

        :param problem: The problem to be optimized.
        :param swarm_size: The size of the swarm.
        :param dim: The dimension of the problem.
        :param max_iter: The maximum number of iterations.
        :param minimize: Whether to minimize or maximize the problem.
        '''
        self.problem = problem
        self.swarm_size = swarm_size
        self.dim = dim
        self.max_iter = max_iter
        # TODO: deal with None max_iter
        # TODO: different possible initializations
        self.swarm = np.random.uniform(self.problem.bounds[:, 0],
                                       self.problem.bounds[:, 1],
                                       (self.swarm_size, self.dim))
        # TODO: different velocity initialization
        self.velocity = np.zeros((self.swarm_size, self.dim))
        self.minimize = minimize
        if not self.minimize:
            self.problem = lambda x: -self.problem(x)
        self.pbest = copy.deepcopy(self.swarm)
        self.pbest_fitness = self.problem(self.pbest)

    def update(self):
        '''
        Update the swarm for one iteration.
        '''
        pass

    def optimize(self):
        '''
        Optimize the problem.
        '''
        pass
