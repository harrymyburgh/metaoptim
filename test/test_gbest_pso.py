import inspect
import sys

from metaoptim.bench_func import Sphere
import metaoptim.config as config
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal


class TestGBestPSOSphere(unittest.TestCase):
    def test_no_accel(self):
        sphere = Sphere(4)
        config._disable_jit = True
        from metaoptim.pso.gbest_pso import GBestPSO
        pso = GBestPSO(sphere, 50, 4, 1000)
        self.assertAlmostEqual(pso.optimize()[1], sphere.optimum_value)
        config._disable_jit = True

    def test_accel(self):
        sphere = Sphere(4)
        config._disable_jit = False
        from metaoptim.pso.gbest_pso import GBestPSO
        pso = GBestPSO(sphere, 50, 4, 1000)
        self.assertAlmostEqual(pso.optimize()[1], sphere.optimum_value)
        config._disable_jit = True

    def test_accel_parallel(self):
        sphere = Sphere(4)
        config._disable_jit = False
        config._numba_parallel = True
        from metaoptim.pso.gbest_pso import GBestPSO
        pso = GBestPSO(sphere, 50, 4, 1000)
        self.assertAlmostEqual(pso.optimize()[1], sphere.optimum_value)
        config._disable_jit = True
        config._numba_parallel = False

    def test_max_dyn_iter_stop(self):
        sphere = Sphere(4)
        config._disable_jit = True
        from metaoptim.pso.gbest_pso import GBestPSO
        pso = GBestPSO(sphere, 50, 4, 100000, 50)
        self.assertAlmostEqual(pso.optimize()[1], sphere.optimum_value)

    def test_max_iter_stop(self):
        sphere = Sphere(4)
        config._disable_jit = True
        from metaoptim.pso.gbest_pso import GBestPSO
        pso = GBestPSO(sphere, 50, 4, None, 50)
        self.assertAlmostEqual(pso.optimize()[1], sphere.optimum_value)
