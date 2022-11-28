from metaoptim.benchmark_functions.bench_func import BenchFunc
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
    def __init__(self, dim, bounds=[-5.12, 5.12], A=10):
        super().__init__(dim)
        self.bounds = np.array([bounds] * dim)
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
    def __init__(self, dim, bounds=[-32.768, 32.768], a=20, b=0.2, c=2 * np.pi):
        super().__init__(dim)
        self.bounds = np.array([bounds] * dim)
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


class StyblinskiTang(VaryingDimBenchFunc):
    def __init__(self, dim, bounds=[-5, 5]):
        super().__init__(dim)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise TypeError("Bounds (bounds) must be a list of length 2.")
        self.bounds = np.array([bounds] * dim)
        self.optimum_value = -39.16616570377142 * dim
        self.name = "StyblinskiTang"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return np.sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2


class Griewank(VaryingDimBenchFunc):
    def __init__(self, dim, bounds=[-600, 600]):
        super().__init__(dim)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise TypeError("Bounds (bounds) must be a list of length 2.")
        self.bounds = np.array([bounds] * dim)
        self.optimum_value = 0.0
        self.name = "Griewank"
        self.range_sqrt = np.sqrt(np.arange(1, self.dim + 1))

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / self.range_sqrt))


class Levy(VaryingDimBenchFunc):
    def __init__(self, dim, bounds=[-10, 10]):
        super().__init__(dim)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise TypeError("Bounds (bounds) must be a list of length 2.")
        self.bounds = np.array([bounds] * dim)
        self.optimum_value = 0.0
        self.name = "Levy"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        w = 1 + (x - 1) / 4
        return np.sin(np.pi * w[0]) ** 2 + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)) + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)


class Schwefel(VaryingDimBenchFunc):
    def __init__(self, dim, bounds=[-500, 500]):
        super().__init__(dim)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise TypeError("Bounds (bounds) must be a list of length 2.")
        self.bounds = np.array([bounds] * dim)
        self.optimum_value = 0.0
        self.name = "Schwefel"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return 418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class Trid(VaryingDimBenchFunc):
    def __init__(self, dim):
        super().__init__(dim)
        self.bounds = np.array([[-(dim ** 2), dim ** 2]] * dim)
        self.optimum_value = -dim * (dim + 4) * (dim - 1) / 6
        self.name = "Trid"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return np.sum((x - 1) ** 2) - np.sum(x[:-1] * x[1:])


class DixonPrice(VaryingDimBenchFunc):
    def __init__(self, dim, bounds=[-10, 10]):
        super().__init__(dim)
        self.bounds = np.array([bounds] * dim)
        self.optimum_value = 0.0
        self.name = "DixonPrice"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return (x[0] - 1) ** 2 + np.sum(2 * (2 * np.arange(1, self.dim) + 1) * (2 * x[1:] ** 2 - x[:-1]) ** 2)
