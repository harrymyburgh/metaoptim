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


class GoldsteinPrice(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-2, 2], [-2, 2]])
        self.optimum_value = 3.0
        self.dim = 2
        self.name = "Goldstein-Price"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * (30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))


class Booth(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-10, 10], [-10, 10]])
        self.optimum_value = 0.0
        self.dim = 2
        self.name = "Booth"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


class BukinN6(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-15, -5], [-3, 3]])
        self.optimum_value = 0.0
        self.dim = 2
        self.name = "Bukin"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)


class Matyas(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-10, 10], [-10, 10]])
        self.optimum_value = 0.0
        self.dim = 2
        self.name = "Matyas"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


class LeviN13(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-10, 10], [-10, 10]])
        self.optimum_value = 0.0
        self.dim = 2
        self.name = "Levi"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return np.sin(3 * np.pi * x[0]) ** 2 + (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2) + (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)


class Himmelblau(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-5, 5], [-5, 5]])
        self.optimum_value = 0.0
        self.dim = 2
        self.name = "Himmelblau"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


class ThreeHumpCamel(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-5, 5], [-5, 5]])
        self.optimum_value = 0.0
        self.dim = 2
        self.name = "Three-Hump Camel"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2


class Easom(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-100, 100], [-100, 100]])
        self.optimum_value = -1.0
        self.dim = 2
        self.name = "Easom"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2)


class CrossInTray(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-10, 10], [-10, 10]])
        self.optimum_value = -2.06261
        self.dim = 2
        self.name = "Cross-in-Tray"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return -0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))) + 1) ** 0.1


class Eggholder(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-512, 512], [-512, 512]])
        self.optimum_value = -959.6407
        self.dim = 2
        self.name = "Eggholder"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0] / 2 + 47))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))


class HolderTable(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-10, 10], [-10, 10]])
        self.optimum_value = -19.2085
        self.dim = 2
        self.name = "Holder Table"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))


class McCormick(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-1.5, 4], [-3, 4]])
        self.optimum_value = -1.9133
        self.dim = 2
        self.name = "McCormick"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1


class SchafferN2(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-100, 100], [-100, 100]])
        self.optimum_value = 0.0
        self.dim = 2
        self.name = "Schaffer N.2"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return 0.5 + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2


class SchafferN4(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-100, 100], [-100, 100]])
        self.optimum_value = 0.292579
        self.dim = 2
        self.name = "Schaffer N.4"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape != (self.dim,):
            raise ValueError("Input (x) must have shape ({},).".format(self.dim))
        return 0.5 + (np.cos(np.sin(np.abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
