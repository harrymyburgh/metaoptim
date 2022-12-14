from metaoptim.pso.pso import PSO
import copy
import numpy as np
import numba
from tqdm import tqdm
import metaoptim.config as config

numba.config.DISABLE_JIT = config._disable_jit
numba.config.NUMBA_NUM_THREADS = config._numba_num_threads


@numba.jit(nopython=config._numba_nopython, cache=config._numba_cache,
           parallel=config._numba_parallel, nogil=config._numba_nogil)
def _update_velocity_helper(w, c1, c2, swarm, velocity, pbest, gbest):
    """
    Helper function for updating the velocity.

    :param w: The inertia weight.
    :param c1: The cognitive parameter.
    :param c2: The social parameter.
    :param swarm: The swarm.
    :param velocity: The velocity.
    :param pbest: The personal best.
    :param gbest: The global best.
    :return: The updated velocity.
    """
    r1 = np.random.uniform(0, 1, swarm.shape)
    r2 = np.random.uniform(0, 1, swarm.shape)
    return w * velocity + c1 * r1 * (pbest - swarm) + c2 * r2 * (gbest - swarm)


@numba.jit(nopython=config._numba_nopython, cache=config._numba_cache,
           parallel=config._numba_parallel, nogil=config._numba_nogil)
def _update_swarm_helper(swarm, velocity):
    '''
    Helper function for updating the swarm.

    :param swarm: The swarm.
    :param velocity: The velocity.
    :return: The updated swarm.
    '''
    return swarm + velocity


@numba.jit(nopython=config._numba_nopython, cache=config._numba_cache,
           parallel=config._numba_parallel, nogil=config._numba_nogil)
def _update_bests_helper(fitness, swarm, pbest, pbest_fitness, gbest,
                         gbest_fitness, minimize):
    """
    Helper function for updating the personal and global bests.

    :param fitness: The fitness of the swarm.
    :param swarm: The swarm.
    :param pbest: The personal best.
    :param pbest_fitness: The fitness of the personal best.
    :param gbest: The global best.
    :param gbest_fitness: The fitness of the global best.
    :param minimize: Whether to minimize or maximize the problem.
    :return: The updated personal best, the updated personal best fitness, the
    updated global best, and the updated global best fitness.
    """
    if minimize:
        pbest_mask = fitness < pbest_fitness
    else:
        pbest_mask = fitness > pbest_fitness
    pbest[pbest_mask] = swarm[pbest_mask]
    pbest_fitness[pbest_mask] = fitness[pbest_mask]
    if minimize:
        best_particle = np.argmin(pbest_fitness)
    else:
        best_particle = np.argmax(pbest_fitness)
    if minimize:
        if pbest_fitness[best_particle] < gbest_fitness:
            gbest = np.copy(pbest[best_particle])
            gbest_fitness = pbest_fitness[best_particle]
    else:
        if pbest_fitness[best_particle] > gbest_fitness:
            gbest = np.copy(pbest[best_particle])
            gbest_fitness = pbest_fitness[best_particle]
    return pbest, pbest_fitness, gbest, gbest_fitness


class GBestPSO(PSO):
    def __init__(self, problem, swarm_size, dim, max_iter, minimize=True,
                 w=0.7298, c1=1.49618, c2=1.49618):
        """
        A class for the global best particle swarm optimization algorithm. This
        particle swarm optimization algorithm uses an inertia weight, cognitive
        parameter, and social parameter.

        :param problem: The problem to be optimized.
        :param swarm_size: The size of the swarm.
        :param dim: The dimension of the swarm.
        :param max_iter: The maximum number of iterations.
        :param minimize: Whether to minimize the problem.
        :param w: The inertia weight.
        :param c1: The cognitive parameter.
        :param c2: The social parameter.
        """
        super().__init__(problem, swarm_size, dim, max_iter, minimize)
        self.gbest = copy.deepcopy(self.swarm[np.argmin(self.pbest_fitness)])
        self.gbest_fitness = self.problem(self.gbest)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        # TODO: velocity clamping

    def update(self):
        """
        Update the swarm for one iteration.
        """
        self._update_bests()
        self._update_velocity()
        self._update_swarm()

    def optimize(self):
        """
        Optimize the problem.
        """
        for _ in tqdm(range(self.max_iter)):
            self.update()
        return self.gbest, self.gbest_fitness

    def _update_bests(self):
        """
        Update the best positions and fitnesses.
        """
        fitness = self.problem(self.swarm)
        self.pbest, self.pbest_fitness, self.gbest, self.gbest_fitness = \
            _update_bests_helper(fitness, self.swarm, self.pbest,
                                 self.pbest_fitness, self.gbest,
                                 self.gbest_fitness, self.minimize)

    def _update_velocity(self):
        """
        Update the velocity of the swarm.
        """
        self.velocity = _update_velocity_helper(self.w, self.c1, self.c2,
                                                self.swarm, self.velocity,
                                                self.pbest, self.gbest)

    def _update_swarm(self):
        '''
        Update the swarm.
        '''
        self.swarm = _update_swarm_helper(self.swarm, self.velocity)
