import numpy as np
cimport numpy as np

cdef class mixture:
    cpdef public np.ndarray[double, ndim = 2, mode = "c"] means
    cpdef public np.ndarray[double, ndim = 3, mode = "c"] covs
    cpdef public np.ndarray[double, ndim = 1, mode = "c"] w
    cpdef public np.ndarray[double, ndim = 1, mode = "c"] log_w
    cpdef public np.ndarray[double, ndim = 2, mode = "c"] bounds
