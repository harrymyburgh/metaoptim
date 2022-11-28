import numpy as np


class BenchFunc:
    def __init__(self):
        pass

    def eval(self, x):
        pass

    def __call__(self, x):
        return self.eval(x)

    def get_bounds(self):
        pass

    def get_optimum_value(self):
        pass

    def get_optimum_point(self):
        pass

    def get_name(self):
        pass

    def get_dim(self):
        pass


class VaryingDimBenchFunc(BenchFunc):
    def __init__(self, dim):
        super().__init__()
        if not isinstance(dim, int):
            raise TypeError("Dimension (dim) must be an integer.")
        if dim <= 0:
            raise ValueError("Dimension (dim) must be greater than 0.")
        self.dim = dim

    def get_dim(self):
        return self.dim


class Rastrigin(VaryingDimBenchFunc):
    def __init__(self, dim, A=10):
        self.bounds = np.array([[-5.12, 5.12]] * dim)
        self.optimum_value = 0.0
        self.optimum_point = np.zeros(dim)
        self.A = A

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return np.sum((x ** 2) - (self.A * np.cos(2 * np.pi * x))) + (self.A * self.dim)

    def get_bounds(self):
        return self.bounds

    def get_optimum_value(self):
        return self.optimum_value

    def get_optimum_point(self):
        return self.optimum_point

    def get_name(self):
        return "Rastrigin"


class Ackley(VaryingDimBenchFunc):
    def __init__(self, dim, a=20, b=0.2, c=2 * np.pi):
        self.bounds = np.array([[-32.768, 32.768]] * dim)
        self.optimum_value = 0.0
        self.optimum_point = np.zeros(dim)
        self.a = a
        self.b = b
        self.c = c

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return -self.a * np.exp(-self.b * np.sqrt(np.sum(x ** 2) / self.dim)) - np.exp((1 / self.dim) * np.sum(np.cos(self.c * x))) + self.a + np.exp(1)

    def get_bounds(self):
        return self.bounds

    def get_optimum_value(self):
        return self.optimum_value

    def get_optimum_point(self):
        return self.optimum_point

    def get_name(self):
        return "Ackley"