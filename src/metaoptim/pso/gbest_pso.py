from metaoptim.bench_func import BenchFunc
from metaoptim.pso.pso import PSO
import copy
import numpy as np
import numba
from tqdm import tqdm
import metaoptim.config as config
from typing import Callable, Optional, Tuple, Union

numba.config.DISABLE_JIT = config._disable_jit
numba.config.NUMBA_NUM_THREADS = config._numba_num_threads


@numba.jit(nopython=config._numba_nopython, cache=config._numba_cache,
           parallel=config._numba_parallel, nogil=config._numba_nogil)
def _update_velocity_helper(w: float, c1: float, c2: float, swarm: np.ndarray, 
                           velocity: np.ndarray, pbest: np.ndarray, gbest: np.ndarray) -> np.ndarray:
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
def _update_swarm_helper(swarm: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    '''
    Helper function for updating the swarm.

    :param swarm: The swarm.
    :param velocity: The velocity.
    :return: The updated swarm.
    '''
    return swarm + velocity


@numba.jit(nopython=config._numba_nopython, cache=config._numba_cache,
           parallel=config._numba_parallel, nogil=config._numba_nogil)
def _update_bests_helper(fitness: np.ndarray, swarm: np.ndarray, pbest: np.ndarray, 
                         pbest_fitness: np.ndarray, gbest: np.ndarray,
                         gbest_fitness: Union[float, np.ndarray], minimize: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[float, np.ndarray]]:
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
    def __init__(self, problem: BenchFunc,
                 swarm_size: int, dim: int, max_iter: Optional[int], 
                 conv_buffer: int = 0, epsilon: float = 1e-10, minimize: bool = True, 
                 w: float = 0.7298, c1: float = 1.49618, c2: float = 1.49618, 
                 verbose: bool = False) -> None:
        """
        A class for the global best particle swarm optimization algorithm. This
        particle swarm optimization algorithm uses an inertia weight, cognitive
        parameter, and social parameter.

        :param problem: The problem to be optimized.
        :param swarm_size: The size of the swarm.
        :param dim: The dimension of the problem.
        :param max_iter: The maximum number of iterations. If set to None,
               ensure the definition of a stopping condition using conv_buffer
               and epsilon. Note that a combination of a maximum number of
               iterations and conv_buffer/epsilon can be used to define the
               stopping condition.
        :param conv_buffer: The buffer size of global best fitness values to be
               approximately equal within the threshold epsilon. It is
               recommended to set this to at least 20.
        :param epsilon: The error for which global best fitness values will be
               considered approximately equal. If
               |f(x^*_{t-1}) - f(x^*_t)| < epsilon, then the global best values
               are considered approximately equal.
        :param minimize: Whether to minimize or maximize the problem.
        :param w: The inertia weight.
        :param c1: The cognitive parameter.
        :param c2: The social parameter.
        :param verbose: Whether to print the progress of the optimization.
        """
        super().__init__(problem, swarm_size, dim, max_iter, conv_buffer,
                         epsilon, minimize, verbose)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        # TODO: velocity clamping

    def update(self) -> None:
        """
        Update the swarm for one iteration.
        """
        self._update_bests()
        self._update_velocity()
        self._update_swarm()

    def optimize(self) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """
        Optimize the problem.
        """
        return super().optimize()

    def _update_bests(self) -> None:
        """
        Update the best positions and fitnesses.
        """
        fitness = self.problem(self.swarm)
        self.pbest, self.pbest_fitness, self.gbest, self.gbest_fitness = \
            _update_bests_helper(fitness, self.swarm, self.pbest,
                                 self.pbest_fitness, self.gbest,
                                 self.gbest_fitness, self.minimize)

    def _update_velocity(self) -> None:
        """
        Update the velocity of the swarm.
        """
        self.velocity = _update_velocity_helper(self.w, self.c1, self.c2,
                                                self.swarm, self.velocity,
                                                self.pbest, self.gbest)

    def _update_swarm(self) -> None:
        '''
        Update the swarm.
        '''
        self.swarm = _update_swarm_helper(self.swarm, self.velocity)
