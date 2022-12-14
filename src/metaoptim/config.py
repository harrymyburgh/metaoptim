import multiprocessing
_numba_cache = True  # Set to True/False to enable/disable Numba cache
_numba_parallel = False  # Set to True/False to enable/disable Numba parallel
_numba_nopython = True  # Set to True/False to enable/disable Numba nopython
_numba_nogil = True  # Set to True/False to enable/disable Numba nogil
# Set to the number of threads to use for Numba
_numba_num_threads = multiprocessing.cpu_count()
_disable_jit = True  # Set to True/False to enable/disable Numba JIT
