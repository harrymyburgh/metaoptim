import inspect
import sys

from metaoptim.bench_func import Beale, GoldsteinPrice, Booth, BukinN6, \
    Matyas, LevyN13, Himmelblau, ThreeHumpCamel, Easom, CrossInTray, \
    Eggholder, McCormick, SchafferN2, SchafferN4, DropWave, Rastrigin, \
    Rosenbrock, Ackley, Sphere, StyblinskiTang, Griewank, Levy, DixonPrice
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal


class TestBeale(unittest.TestCase):
    def test(self):
        beale = Beale()
        x = np.array([3, 0.5])
        self.assertAlmostEqual(beale(x), 0.0)

    def test_multiple(self):
        beale = Beale()
        x = np.array([[3, 0.5], [3, 0.5]])
        assert_array_almost_equal(beale(x), np.array([0.0, 0.0]))


class TestGoldsteinPrice(unittest.TestCase):
    def test(self):
        goldstein_price = GoldsteinPrice()
        x = np.array([0, -1])
        self.assertAlmostEqual(goldstein_price(x), 3.0)

    def test_multiple(self):
        goldstein_price = GoldsteinPrice()
        x = np.array([[0, -1], [0, -1]])
        assert_array_almost_equal(goldstein_price(x), np.array([3.0, 3.0]))


class TestBooth(unittest.TestCase):
    def test(self):
        booth = Booth()
        x = np.array([1, 3])
        self.assertAlmostEqual(booth(x), 0.0)

    def test_multiple(self):
        booth = Booth()
        x = np.array([[1, 3], [1, 3]])
        assert_array_almost_equal(booth(x), np.array([0.0, 0.0]))


class TestBukinN6(unittest.TestCase):
    def test(self):
        bukin_n6 = BukinN6()
        x = np.array([-10, 1])
        self.assertAlmostEqual(bukin_n6(x), 0.0)

    def test_multiple(self):
        bukin_n6 = BukinN6()
        x = np.array([[-10, 1], [-10, 1]])
        assert_array_almost_equal(bukin_n6(x), np.array([0.0, 0.0]))


class TestMatyas(unittest.TestCase):
    def test(self):
        matyas = Matyas()
        x = np.array([0, 0])
        self.assertAlmostEqual(matyas(x), 0.0)

    def test_multiple(self):
        matyas = Matyas()
        x = np.array([[0, 0], [0, 0]])
        assert_array_almost_equal(matyas(x), np.array([0.0, 0.0]))


class TestLevyN13(unittest.TestCase):
    def test(self):
        levy_n13 = LevyN13()
        x = np.array([1, 1])
        self.assertAlmostEqual(levy_n13(x), 0.0)

    def test_multiple(self):
        levy_n13 = LevyN13()
        x = np.array([[1, 1], [1, 1]])
        assert_array_almost_equal(levy_n13(x), np.array([0.0, 0.0]))


class TestHimmelblau(unittest.TestCase):
    def test(self):
        himmelblau = Himmelblau()
        x = np.array([3, 2])
        self.assertAlmostEqual(himmelblau(x), 0.0)

    def test_multiple(self):
        himmelblau = Himmelblau()
        x = np.array([[3, 2], [3, 2]])
        assert_array_almost_equal(himmelblau(x), np.array([0.0, 0.0]))


class TestThreeHumpCamel(unittest.TestCase):
    def test(self):
        thc = ThreeHumpCamel()
        x = np.array([0, 0])
        self.assertAlmostEqual(thc(x), 0.0)

    def test_multiple(self):
        thc = ThreeHumpCamel()
        x = np.array([[0, 0], [0, 0]])
        assert_array_almost_equal(thc(x), np.array([0.0, 0.0]))


class TestEasom(unittest.TestCase):
    def test(self):
        easom = Easom()
        x = np.array([np.pi, np.pi])
        self.assertAlmostEqual(easom(x), -1.0)

    def test_multiple(self):
        easom = Easom()
        x = np.array([[np.pi, np.pi], [np.pi, np.pi]])
        assert_array_almost_equal(easom(x), np.array([-1.0, -1.0]))


class TestCrossInTray(unittest.TestCase):
    def test(self):
        cit = CrossInTray()
        x = np.array([1.34941, 1.34941])
        self.assertAlmostEqual(cit(x), -2.06261, places=5)

    def test_multiple(self):
        cit = CrossInTray()
        x = np.array([[1.34941, 1.34941], [1.34941, 1.34941]])
        assert_array_almost_equal(cit(x), np.array([-2.06261, -2.06261]),
                                  decimal=5)


class TestEggholder(unittest.TestCase):
    def test(self):
        eggholder = Eggholder()
        x = np.array([512, 404.2319])
        self.assertAlmostEqual(eggholder(x), -959.6407, places=4)

    def test_multiple(self):
        eggholder = Eggholder()
        x = np.array([[512, 404.2319], [512, 404.2319]])
        assert_array_almost_equal(eggholder(x),
                                  np.array([-959.6407, -959.6407]), decimal=4)


class TestMcCormick(unittest.TestCase):
    def test(self):
        mccormick = McCormick()
        x = np.array([-0.54719, -1.54719])
        self.assertAlmostEqual(mccormick(x), -1.9133, places=3)

    def test_multiple(self):
        mccormick = McCormick()
        x = np.array([[-0.54719, -1.54719], [-0.54719, -1.54719]])
        assert_array_almost_equal(mccormick(x), np.array([-1.9133, -1.9133]),
                                  decimal=4)


class TestSchafferN2(unittest.TestCase):
    def test(self):
        schaffer_n2 = SchafferN2()
        x = np.array([0, 0])
        self.assertAlmostEqual(schaffer_n2(x), 0.0)

    def test_multiple(self):
        schaffer_n2 = SchafferN2()
        x = np.array([[0, 0], [0, 0]])
        assert_array_almost_equal(schaffer_n2(x), np.array([0.0, 0.0]))


class TestSchafferN4(unittest.TestCase):
    def test(self):
        schaffer_n4 = SchafferN4()
        x = np.array([0, 1.25313])
        self.assertAlmostEqual(schaffer_n4(x), 0.292579, places=3)

    def test_multiple(self):
        schaffer_n4 = SchafferN4()
        x = np.array([[0, 1.25313], [0, 1.25313]])
        assert_array_almost_equal(schaffer_n4(x),
                                  np.array([0.292579, 0.292579]), decimal=3)


class TestDropWave(unittest.TestCase):
    def test(self):
        drop_wave = DropWave()
        x = np.array([0, 0])
        self.assertAlmostEqual(drop_wave(x), -1.0)

    def test_multiple(self):
        drop_wave = DropWave()
        x = np.array([[0, 0], [0, 0]])
        assert_array_almost_equal(drop_wave(x), np.array([-1.0, -1.0]))


class TestRastrigin(unittest.TestCase):
    def test(self):
        rastrigin = Rastrigin(dim=2)
        x = np.array([0, 0])
        self.assertAlmostEqual(rastrigin(x), 0.0)

    def test_multiple(self):
        rastrigin = Rastrigin(dim=2)
        x = np.array([[0, 0], [0.0, 0.0]])
        assert_array_almost_equal(rastrigin(x), np.array([0.0, 0.0]))

    def test_high_dim(self):
        rastrigin = Rastrigin(dim=100)
        x = np.zeros(100)
        self.assertAlmostEqual(rastrigin(x), 0.0)


class TestRosenbrock(unittest.TestCase):
    def test(self):
        rosenbrock = Rosenbrock(dim=2)
        x = np.array([1, 1])
        self.assertAlmostEqual(rosenbrock(x), 0.0)

    def test_multiple(self):
        rosenbrock = Rosenbrock(dim=2)
        x = np.array([[1, 1], [1.0, 1.0]])
        assert_array_almost_equal(rosenbrock(x), np.array([0.0, 0.0]))

    def test_high_dim(self):
        rosenbrock = Rosenbrock(dim=100)
        x = np.ones(100)
        self.assertAlmostEqual(rosenbrock(x), 0.0)


class TestAckley(unittest.TestCase):
    def test(self):
        ackley = Ackley(dim=2)
        x = np.array([0, 0])
        self.assertAlmostEqual(ackley(x), 0.0)

    def test_multiple(self):
        ackley = Ackley(dim=2)
        x = np.array([[0, 0], [0.0, 0.0]])
        assert_array_almost_equal(ackley(x), np.array([0.0, 0.0]))

    def test_high_dim(self):
        ackley = Ackley(dim=100)
        x = np.zeros(100)
        self.assertAlmostEqual(ackley(x), 0.0)


class TestSphere(unittest.TestCase):
    def test(self):
        sphere = Sphere(dim=2)
        x = np.array([0, 0])
        self.assertAlmostEqual(sphere(x), 0.0)

    def test_multiple(self):
        sphere = Sphere(dim=2)
        x = np.array([[0, 0], [0.0, 0.0]])
        assert_array_almost_equal(sphere(x), np.array([0.0, 0.0]))

    def test_high_dim(self):
        sphere = Sphere(dim=100)
        x = np.zeros(100)
        self.assertAlmostEqual(sphere(x), 0.0)


class TestStyblinskiTang(unittest.TestCase):
    def test(self):
        styblinski_tang = StyblinskiTang(dim=2)
        x = np.array([-2.903534, -2.903534])
        self.assertAlmostEqual(styblinski_tang(x), -78.332, places=3)

    def test_multiple(self):
        styblinski_tang = StyblinskiTang(dim=2)
        x = np.array([[-2.903534, -2.903534], [-2.903534, -2.903534]])
        assert_array_almost_equal(styblinski_tang(x),
                                  np.array([-78.332, -78.332]), decimal=3)

    def test_high_dim(self):
        styblinski_tang = StyblinskiTang(dim=100)
        x = np.ones(100) * -2.903534
        self.assertAlmostEqual(styblinski_tang(x),
                               styblinski_tang.get_optimum_value(), places=3)


class TestGriewank(unittest.TestCase):
    def test(self):
        griewank = Griewank(dim=2)
        x = np.array([0, 0])
        self.assertAlmostEqual(griewank(x), 0.0)

    def test_multiple(self):
        griewank = Griewank(dim=2)
        x = np.array([[0, 0], [0.0, 0.0]])
        assert_array_almost_equal(griewank(x), np.array([0.0, 0.0]))

    def test_high_dim(self):
        griewank = Griewank(dim=100)
        x = np.zeros(100)
        self.assertAlmostEqual(griewank(x), 0.0)


class TestLevy(unittest.TestCase):
    def test(self):
        levy = Levy(dim=2)
        x = np.array([1, 1])
        self.assertAlmostEqual(levy(x), 0.0)

    def test_multiple(self):
        levy = Levy(dim=2)
        x = np.array([[1, 1], [1.0, 1.0]])
        assert_array_almost_equal(levy(x), np.array([0.0, 0.0]))

    def test_high_dim(self):
        levy = Levy(dim=100)
        x = np.ones(100)
        self.assertAlmostEqual(levy(x), 0.0)


# TODO test DixonPrice
# TODO test Trid
# TODO test Schwefel
# TODO test Shubert

class TestAll(unittest.TestCase):
    def test(self):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj) and issubclass(obj, unittest.TestCase):
                suite = unittest.TestLoader().loadTestsFromTestCase(obj)
                unittest.TextTestRunner(verbosity=2).run(suite)
