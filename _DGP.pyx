import cython
import numpy as np
cimport numpy as np
# import C standard library: https://en.wikipedia.org/wiki/C_standard_library
from libc.stdlib cimport srand, rand, RAND_MAX  # RAND_MAX = 2147483647
from libc.math cimport pow, abs, log, sqrt, exp, M_E as e


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double rand_unif():
    cdef:
        double u;
    
    u = rand() / (RAND_MAX + 1.0)
    return u


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double test_rand_unif(long int N):
    cdef:
        double u;
        long int i;
    for i in range(N):
        u = rand_unif()
    return u


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double rand_exp(double lamb):
    """ Inverse-transform method """
    cdef:
        double u;
    
    u = rand_unif()
    return -log(1- u) / lamb


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double rand_normal():
    """ Box-muller method """
    cdef double x1, x2, w

    w = 2.0
    while w >= 1.0:
        x1 = 2.0 * rand_unif() - 1.0
        x2 = 2.0 * rand_unif() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef int rand_poisson(double lamb):
    """ Inversion method """
    cdef:
        int x = 0
        double p = exp(-lamb), s = p
        double u = rand_unif()

    while u > s:
        x += 1
        p *= lamb/x
        s += p

    return x


# The following is deprecated because it is 5x slower than the inversion method.
#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.cdivision(True)
#cpdef double rand_poisson_knuth(double lamb):
#    """Knuth method"""
#    cdef:
#        double L = exp(-lamb), k = 0, p = 1

#    while True: 
#        k += 1
#        p *= rand_unif()
#        if p > L: continue 
#        break

#    return k - 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef int rand_zt_poisson(double lamb):
    """ Zero-truncated poisson """
    cdef:
        int x = rand_poisson(lamb)
    
    if lamb <= 0:
        return 0
    
    while x == 0:
        x = rand_poisson(lamb)

    return x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] gen_CIR(double a, double[:] bs, double sigma, double r_0, 
                        double[:] ts, long int seed):
    """ Generate CIR process.
    $$
        dr_t = a(b-r_t)dt + \sigma\sqrt{r_t}dW_t 
    $$
    Reference: python package `tick`
    """
    srand(seed)
    cdef:
        long int N = len(ts)
        double dt = (np.max(ts) - np.min(ts))/N
        double[:] dW_t = np.random.randn(N)*np.sqrt(dt)
        double[:] r_t = np.zeros(N)
        long int i = 0
        
    assert r_0 >= 0
    
    r_t[0] = r_0 
    for i in range(1,N):
        r_t[i] = r_t[i-1] + a*(bs[i-1]-r_t[i-1])*dt + sigma*sqrt(r_t[i-1])*dW_t[i]
        
    return r_t
