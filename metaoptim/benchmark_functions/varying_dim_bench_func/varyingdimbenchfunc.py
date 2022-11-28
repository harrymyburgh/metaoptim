from metaoptim.benchmark_functions.benchfunc import BenchFunc
import numpy as np


class VaryingDimBenchFunc(BenchFunc):
    def __init__(self, dim):
        super().__init__()
        if not isinstance(dim, int):
            raise TypeError("Dimension (dim) must be an integer.")
        if dim <= 1:
            raise ValueError("Dimension (dim) must be greater than 1.")
        self.dim = dim


class Rastrigin(VaryingDimBenchFunc):
    def __init__(self, dim, A=10):
        super().__init__(dim)
        self.bounds = np.array([[-5.12, 5.12]] * dim)
        self.optimum_value = 0.0
        self.A = A
        self.name = "Rastrigin"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return np.sum((x ** 2) - (self.A * np.cos(2 * np.pi * x))) + (self.A * self.dim)


class Ackley(VaryingDimBenchFunc):
    def __init__(self, dim, a=20, b=0.2, c=2 * np.pi):
        super().__init__(dim)
        self.bounds = np.array([[-32.768, 32.768]] * dim)
        self.optimum_value = 0.0
        self.a = a
        self.b = b
        self.c = c
        self.name = "Ackley"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return -self.a * np.exp(-self.b * np.sqrt(np.sum(x ** 2) / self.dim)) - np.exp((1 / self.dim) * np.sum(np.cos(self.c * x))) + self.a + np.exp(1)


class Sphere(VaryingDimBenchFunc):
    def __init__(self, dim, bounds=[-5.12, 5.12]):
        super().__init__(dim)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise TypeError("Bounds (bounds) must be a list of length 2.")
        self.bounds = np.array([bounds] * dim)
        self.optimum_value = 0.0
        self.name = "Sphere"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return np.sum(x ** 2)


class Rosenbrock(VaryingDimBenchFunc):
    def __init__(self, dim, bounds=[-10, 10]):
        super().__init__(dim)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise TypeError("Bounds (bounds) must be a list of length 2.")
        self.bounds = np.array([bounds] * dim)
        self.optimum_value = 0.0
        self.name = "Rosenbrock"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)