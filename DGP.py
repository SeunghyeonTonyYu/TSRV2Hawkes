import numpy as np
from tick.base import TimeFunction
from tick.hawkes import SimuHawkes, SimuInhomogeneousPoisson, \
                SimuHawkesExpKernels, HawkesKernelExp, HawkesKernelPowerLaw

from _DGP import *


def gen_U_shape_CIR(t, seed, sigma,
                    a=30, power=4, #sigma=1.5
                    x_offset=.53, y_offset=1/24, scale=20,#32
                   ):
    
    f  = scale*((t - x_offset)**power + y_offset)
    f0 = scale*((0 - x_offset)**power + y_offset)
    f_dot = scale*(power*(t - x_offset)**(power-1))
    b = f_dot/a + f
    
    X = gen_CIR(a, b, sigma, f0, t, seed) + np.zeros(len(t))
    
    return X


def gen_mu_sigma2(t, typ, seed):
    
    np.random.seed(seed)
    
    if typ == 'flat':
        mu = np.repeat(1., len(t))
        sigma2 = 0
        return mu, sigma2
    
    elif typ == 'U-shape':
        mu = np.array([-1])
        sigma = 0
        while any(mu < 0):
            mu = gen_U_shape_CIR(t, np.random.randint(10000000), sigma=sigma)
        sigma2 = mu*sigma**2
        return mu, sigma2
    
    elif typ == 'CIR':
        mu = np.array([-1])
        sigma = 3 # 1.5 would be more realistic, but IV test result is aweful.
        while any(mu < 0):
            mu = gen_U_shape_CIR(t, np.random.randint(10000000), sigma=sigma)
        sigma2 = mu*sigma**2
        return mu, sigma2
    
    else:
        raise NotImplementedError("Not implemented for type '%s'"%typ)


def get_burst(mu, seed, 
              pois_lamb=2, width=1/(6.5*3600)):
    
    np.random.seed(seed)
    n_burst = np.random.poisson(pois_lamb)
    if n_burst == 0:
        return 0
    
    mean = np.sum(mu) / len(mu)
    burst_intensity = np.repeat(0, len(mu))
    for _ in range(n_burst):
        burst_t_i = np.random.randint(0, len(mu))
        start = max(burst_t_i - int(width / 2 * len(mu)), 0)
        end   = min(burst_t_i + int(width / 2 * len(mu)), len(mu))
        burst_intensity[start:end] = max(np.random.normal(mean*200, mean*50),
                                         mean*50)
        
    return burst_intensity


def gen_Cox(t, mu, seed, n=150000):
    lambda_t = n * mu
    cox = SimuInhomogeneousPoisson([TimeFunction([t, lambda_t])], end_time=1., 
                                   seed=seed, verbose=False)
    cox.simulate()
    points = cox.timestamps[0]
    del cox
    
    return points


def gen_HawkesExpKernels(t, mu, seed,
                         alpha=1, beta=1, n=150000,
                         force_simulation=False):
    
    baseline_ts = n * mu
    hawkes = SimuHawkesExpKernels([[alpha/beta]], [[n*beta]],
                                  baseline_ts.reshape((1,-1)), 
                                  end_time=1., verbose=False, seed=seed, 
                                  period_length=1,
                                  force_simulation=force_simulation)
    hawkes.simulate()
    points = hawkes.timestamps[0]
    del hawkes
    
    return points


def gen_HawkesPowerKernels(t, mu, seed,
                           multiplier=1, exponent=3, cutoff=1, support=-1, 
                           n=150000, force_simulation=False):

    baseline_ts = TimeFunction([t, n*mu])
    hawkes = SimuHawkes(
        kernels=[[HawkesKernelPowerLaw(multiplier*n**(1-exponent), cutoff/n, 
                                       exponent, support)]],
        baseline=[baseline_ts], seed=seed, verbose=False, 
        force_simulation=force_simulation)
    hawkes.end_time = 1
    hawkes.simulate()
    points = hawkes.timestamps[0]
    del hawkes
    
    return points