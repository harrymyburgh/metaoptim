import copy
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Tuple

from metaoptim.bench_func import BenchFunc


class PSO:
    def __init__(self, problem: BenchFunc,
                 swarm_size: int, dim: int, max_iter: Optional[int], 
                 conv_buffer: int = 0, epsilon: float = 1e-10, 
                 minimize: bool = True, verbose: bool = False) -> None:
        '''
        Initialize the PSO algorithm.

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
        :param verbose: Whether to print the progress of the optimization.
        '''
        self.problem = problem
        self.swarm_size = swarm_size
        self.dim = dim
        self.max_iter = max_iter
        self.conv_buffer = conv_buffer
        self.epsilon = epsilon
        if self.max_iter is None and self.conv_buffer == 0:
            raise AttributeError("No stopping condition set for PSO model.")
        if self.max_iter is not None and self.conv_buffer > 0:
            self.optimization_alg = self._max_dyn_iter_optimize
        elif self.max_iter is None:
            self.optimization_alg = self._dyn_iter_optimize
        else:
            self.optimization_alg = self._max_iter_optimize
        self.countdown = conv_buffer
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
        if self.minimize:
            self.gbest = copy.deepcopy(
                self.swarm[np.argmin(self.pbest_fitness)])
        else:
            self.gbest = copy.deepcopy(
                self.swarm[np.argmax(self.pbest_fitness)])
        self.gbest_fitness = self.problem(self.gbest)
        self.prev_gbest_fitness = float("inf") if self.minimize else -float("inf")

        self.verbose = verbose

    def update(self) -> None:
        '''
        Update the swarm for one iteration.
        '''
        pass

    def optimize(self) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        '''
        Optimize the problem.
        '''
        return self.optimization_alg()

    def _max_iter_optimize(self) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        for _ in tqdm(range(self.max_iter), disable=not self.verbose):
            self.update()
        return self.gbest, self.gbest_fitness

    def _dyn_iter_optimize(self) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        while not tqdm(self.countdown_check()):  # TODO: Fix tqdm
            self.update()
        return self.gbest, self.gbest_fitness

    def _max_dyn_iter_optimize(self) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        for _ in tqdm(range(self.max_iter), disable=not self.verbose):
            if self.countdown_check():
                if self.verbose:
                    print(f"Global best value did not significantly change "
                          f"after {self.conv_buffer} iterations.\nHalting "
                          f"optimization.")
                break
            else:
                self.update()
        return self.gbest, self.gbest_fitness

    def countdown_check(self) -> bool:
        if np.abs(self.prev_gbest_fitness - self.gbest_fitness) < self.epsilon:
            self.countdown -= 1
        else:
            self.countdown = self.conv_buffer
        self.prev_gbest_fitness = self.gbest_fitness
        return self.countdown <= 0
