import numpy as np


class BenchFunc:
    def __init__(self):
        self.bounds = None
        self.optimum_value = None
        self.name = None
        self.dim = None

    def eval(self, x):
        pass

    def __call__(self, x):
        return self.eval(x)

    def get_bounds(self):
        return self.bounds

    def get_optimum_value(self):
        return self.optimum_value

    def get_name(self):
        return self.name

    def get_dim(self):
        return self.dim


class Beale(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-4.5, 4.5], [-4.5, 4.5]])
        self.optimum_value = 0.0
        self.dim = 2
        self.name = "Beale"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2