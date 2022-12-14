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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return (1.5 - x[0] + x[0] * x[1]) ** 2 + (
                2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
                       2.625 - x[0] + x[0] * x[1] ** 3) ** 2


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return (1 + (x[0] + x[1] + 1) ** 2 * (
                19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[
                    1] + 3 * x[1] ** 2)) * (30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                        18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0]
                        * x[1] + 27 * x[1] ** 2))


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


class BukinN6(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-15, -5], [-3, 3]])
        self.optimum_value = 0.0
        self.dim = 2
        self.name = "Bukin N.6"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(
            x[0] + 10)


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


class LevyN13(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-10, 10], [-10, 10]])
        self.optimum_value = 0.0
        self.dim = 2
        self.name = "Levi N.13"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return np.sin(3 * np.pi * x[0]) ** 2 + (x[0] - 1) ** 2 * (
                1 + np.sin(3 * np.pi * x[1]) ** 2) + (x[1] - 1) ** 2 * (
                       1 + np.sin(2 * np.pi * x[1]) ** 2)


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 \
            / 6 + x[0] * x[1] + x[1] ** 2


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(
            -(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2)


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return -0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(
            np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))) + 1) ** 0.1


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0] / 2 + 47))) \
               - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(
            np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[
            1] + 1


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return 0.5 + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (
                1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return 0.5 + (np.cos(
            np.sin(np.abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5) / (
                       1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2


class DropWave(BenchFunc):
    def __init__(self):
        super().__init__()
        self.bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])
        self.optimum_value = -1.0
        self.dim = 2
        self.name = "Drop-Wave"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return -(1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))) / \
            (0.5 * (x[0] ** 2 + x[1] ** 2) + 2)


# TODO: Fix to work with broadcasting
# class Shubert(BenchFunc):
#     def __init__(self):
#         super().__init__()
#         self.bounds = np.array([[-10, 10], [-10, 10]])
#         self.optimum_value = -186.7309
#         self.dim = 2
#         self.name = "Shubert"
#         self.five_range = np.arange(1, 6)
#
#     def eval(self, x):
#         if not isinstance(x, np.ndarray):
#             raise TypeError("Input (x) must be a numpy array.")
#         if x.shape[-1] != self.dim:
#             raise ValueError("Input (x) must
#             have shape ({},).".format(self.dim))
#         x = np.transpose(x)
#         return np.sum(self.five_range * np.cos((self.five_range + 1) * x[0] +
#         self.five_range)) * np.sum(self.five_range *
#         np.cos((self.five_range + 1) * x[1] + self.five_range))


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return np.sum((x ** 2) - (self.A * np.cos(2 * np.pi * x)), axis=0) + (
                self.A * self.dim)


class Ackley(VaryingDimBenchFunc):
    def __init__(self, dim, bounds=[-32.768, 32.768], a=20, b=0.2,
                 c=2 * np.pi):
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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return -self.a * np.exp(
            -self.b * np.sqrt(np.sum(x ** 2, axis=0) / self.dim)) - np.exp(
            (1 / self.dim) * np.sum(np.cos(self.c * x),
                                    axis=0)) + self.a + np.exp(1)


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return np.sum(x ** 2, axis=0)


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2,
                      axis=0)


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=0) / 2


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return 1 + np.sum(x ** 2, axis=0) / 4000 - np.prod(
            np.cos(x / self.range_sqrt), axis=0)


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
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        w = 1 + (x - 1) / 4
        return np.sin(np.pi * w[0]) ** 2 + np.sum(
            (w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2),
            axis=0) + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)


# TODO: Fix Schwefel
# class Schwefel(VaryingDimBenchFunc):
#     def __init__(self, dim, bounds=[-500, 500]):
#         super().__init__(dim)
#         if not isinstance(bounds, list) or len(bounds) != 2:
#             raise TypeError("Bounds (bounds) must be a list of length 2.")
#         self.bounds = np.array([bounds] * dim)
#         self.optimum_value = 0.0
#         self.name = "Schwefel"
#
#     def eval(self, x):
#         if not isinstance(x, np.ndarray):
#             raise TypeError("Input (x) must be a numpy array.")
#         if x.shape[-1] != self.dim:
#             raise ValueError("Input (x) must
#             have shape ({},).".format(self.dim))
#         x = np.transpose(x)
#         return 418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))),
#         axis=0)


# TODO: Fix Trid
# class Trid(VaryingDimBenchFunc):
#     def __init__(self, dim):
#         super().__init__(dim)
#         self.bounds = np.array([[-(dim ** 2), dim ** 2]] * dim)
#         self.optimum_value = -dim * (dim + 4) * (dim - 1) / 6
#         self.name = "Trid"
#
#     def eval(self, x):
#         if not isinstance(x, np.ndarray):
#             raise TypeError("Input (x) must be a numpy array.")
#         if x.shape[-1] != self.dim:
#             raise ValueError("Input (x) must have shape
#             ({},).".format(self.dim))
#         x = np.transpose(x)
#         return np.sum((x - 1) ** 2, axis=0) - np.sum(x[:-1] * x[1:], axis=0)


class DixonPrice(VaryingDimBenchFunc):
    def __init__(self, dim, bounds=[-10, 10]):
        super().__init__(dim)
        self.bounds = np.array([bounds] * dim)
        self.optimum_value = 0.0
        self.name = "DixonPrice"

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input (x) must be a numpy array.")
        if x.shape[-1] != self.dim:
            raise ValueError(
                "Input (x) must have shape ({},).".format(self.dim))
        x = np.transpose(x)
        return (x[0] - 1) ** 2 + np.sum(2 * (2 * np.arange(1, self.dim) + 1)
                                        * (2 * x[1:] ** 2 - x[:-1]) ** 2,
                                        axis=0)
