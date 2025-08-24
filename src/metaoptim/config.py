"""
Configuration module for metaoptim library.
Contains Numba-related configuration variables for optimization performance.
"""

import multiprocessing

# Numba configuration variables with type annotations
_numba_cache: bool = True  # Set to True/False to enable/disable Numba cache
_numba_parallel: bool = False  # Set to True/False to enable/disable Numba parallel
_numba_nopython: bool = True  # Set to True/False to enable/disable Numba nopython
_numba_nogil: bool = True  # Set to True/False to enable/disable Numba nogil
# Set to the number of threads to use for Numba
_numba_num_threads: int = multiprocessing.cpu_count()
_disable_jit: bool = True  # Set to True/False to enable/disable Numba JIT
