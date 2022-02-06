import numpy as np
cimport numpy as np

cdef class mixture_cython:
    cdef public double[:,:]   means
    cdef public double[:,:,:] covs
    cdef public double[:]     w
    cdef public double[:]     log_w
    cdef public double[:,:]   bounds
