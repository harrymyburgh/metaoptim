# METAOPTIM

metaoptim is a Python package for metaheuristic optimization algorithms. It also features various benchmark functions that can be used to test the performance of the algorithms.

## Installation
Ensure that you have Python 3.9 or higher installed. Then, install the package using pip:

```bash
pip3 install metaoptim
```

## Usage

### Configuration

This package has the ability to accelerate the metaheuristic algorithms using `numba`. To control the use of `numba` you can control the following flags in the `metaoptim.config` module:

```python
import multiprocessing
_numba_cache = True                                 # Set to True/False to enable/disable Numba cache
_numba_parallel = False                             # Set to True/False to enable/disable Numba parallel
_numba_nopython = True                              # Set to True/False to enable/disable Numba nopython
_numba_nogil = True                                 # Set to True/False to enable/disable Numba nogil
_numba_num_threads = multiprocessing.cpu_count()    # Set to the number of threads to use for Numba
_disable_jit = True                                 # Set to True/False to enable/disable Numba JIT
```

The aforementioned flags are set to their default values. You can change them to your liking by importing the `config` module and setting the flags to the desired values.
e.g.:

```python
import metaoptim.config as config
config._numba_cache = False
```

However, **please ensure that you import any metaheuristic algorithms after you have set the flags. Otherwise, the flags will not take effect**.

Finally, GPU acceleration is currently not supported, but will be added in the future.

#### Recommendations

Numba is a great tool for accelerating Python code. However, it is not always the best choice. For example, if you are using a metaheuristic algorithm that is already fast, then you may not want to use Numba. In such cases, you can disable Numba by setting the `_disable_jit` flag to `True`.
You should use Numba acceleration when the metaheuristic algorithm is slow, working in high dimensions, or when you are using a large population size.

### Algorithms

The package currently features the following algorithms:

- Global Best Particle Swarm Optimization

Please refer to the class docstrings for more information.

### Benchmark Functions

- Ackley (Dimension: N)
- Beale (Dimension: 2)
- Booth (Dimension: 2)
- Bukin N. 6 (Dimension: 2)
- Cross-in-Tray (Dimension: 2)
- Dixon-Price (Dimension: N)
- Drop-Wave (Dimension: 2)
- Easom (Dimension: 2)
- Eggholder (Dimension: 2)
- Goldstein-Price (Dimension: 2)
- Griewank (Dimension: N)
- Himmelblau (Dimension: 2)
- Holder Table (Dimension: 2)
- Levy (Dimension: N)
- Levy N. 13 (Dimension: 2)
- Matyas (Dimension: 2)
- McCormick (Dimension: 2)
- Rastrigin (Dimension: N)
- Rosenbrock (Dimension: N)
- Schaffer N. 2 (Dimension: 2)
- Schaffer N. 4 (Dimension: 2)
- Sphere (Dimension: N)
- Styblinski-Tang (Dimension: N)
- Three-Hump Camel (Dimension: 2)

### Package Structure

The package is structured as follows:

- `metaoptim`: The package directory.
    - `config.py`: The configuration file.
    - `benchmark_functions`: The benchmark functions package.
      - `bench_func.py`: Contains the `BenchFunc` class.
      - `varying_dim_bench_func`: Variable-dimensional benchmark functions package.
        - `varying_dim_bench_func.py` Variable-dimensional benchmark functions are defined in here and are subclasses of the `VaryingDimBenchFunc` class which is a subclass of the `BenchFunc` class.
    - `metaheuristics`: The metaheuristic algorithms package.
      - `pso`: The Particle Swarm Optimization algorithms package.
        - `pso.py`: The `PSO` parent class.
        - `gbest_pso.py`: The Global Best Particle Swarm Optimization algorithm (`GBestPSO`). Features an inertia weight, a cognitive and a social component.