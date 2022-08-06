__author__ = 'Seunghyeon Yu'

import numpy as np
import pandas as pd


def get_trunc(hat_lamb, Delta_n, 
              C=1, recursion=3):
    
    if len(hat_lamb) < 2:
        return np.repeat(True, len(hat_lamb) - 1)
    
    diff  = hat_lamb[1:] - hat_lamb[:-1]
    tr_lv = C * np.sqrt(np.mean(diff[diff!=0]**2)) * Delta_n**(-1/4)
    mask  = np.abs(diff) < tr_lv

    # Eliminate consecutive intensity burst.
    trunced = hat_lamb[1:][mask]
    diff1   = trunced - np.concatenate(([trunced[0]], trunced[:-1]))
    diff2   = np.concatenate((trunced[1:], [trunced[-1]])) - trunced
    mask2   = np.logical_or(np.abs(diff1) < tr_lv, 
                            np.abs(diff2) < tr_lv)
    for i in np.nonzero(~mask)[0]:
        mask2 = np.insert(mask2, i, False)
    
    # Since the truncation level is biased upward by the burst, truncate again.
    if recursion:
        new_hat_lamb = np.concatenate(([hat_lamb[0]] if mask2[0] else [0], 
                                       diff * mask2)).cumsum()
        mask3 = get_trunc(new_hat_lamb, Delta_n, C=C, recursion=recursion-1)
    else:
        return mask2
        
    return np.logical_and(mask2, mask3)


def TSRV2Hawkes(points, 
                Delta_n=None, T=1, c=.5,
                truncation=True, recursion=True, debug=False):
    """Nonparametric Hawkes estimation using TSRV method
    
    This implementation is the nonparametric estimator under the time-varying
    baseline described in the paper `Nonparametric Estimation of Hawkes 
    Branching Ratio under Time-arying Baseline Intensity` by Yu and Potiron 
    (2022, Preprint).

    Parameters
    ----------
    points : np.array or list
        A collection of time points of the data. It should be 1-dimensional 
        (univariate).
    Delta_n : float, optional
        A length of the localization interval. If it is `None`, it 
        automatically set to be one over square root of the number of `points`,
        $1/\sqrt{N}$.
    T : float, optional
        End time of the `points`. 
    c : float, optional
        The coefficient for determining the length of interval `Delta_n`.
    truncation : Boolean or float, optional
        Truncate intensity burst when it is True. If it is float, it 
        determinies the coefficient of truncation $C$.
    recursion : Boolean, optional
        Based on primary BR estimation, choose appropriate value of `c`.
        When estiamted branching ratio > 0.7, estimate BR again with `c=2` 
        (This is due to the high BR case requires longer Delta_n).
        When estimated branching ratio is closed to zero, use `c=2`.
    """
    return BR_estimation(points=points,
                         Delta_n=Delta_n, T=T, c=c,
                         truncation=truncation, recursion=recursion, 
                         debug=debug)


def BR_estimation(points, 
                  Delta_n=None, T=1, c=.5,
                  truncation=True, recursion=True, debug=False):
    points = np.array(points)
    
    if len(points) <= 4:
        return pd.Series({'N'   : np.nan,     'Delta_n' : np.nan,
                          'Mean': np.nan,     'SD(Mean)': np.nan,
                          'IV'  : np.nan,     'SD(IV)'  : np.nan,
                          'BR'  : np.nan,     'SD(BR)'  : np.nan,
                          'BR_const': np.nan, 'SD(BR_const)': np.nan,
                          'BR - BR_const': np.nan,
                          'SD(BR - BR_const)': np.nan,
                         })
    
    assert len(points.shape) == 1, \
        "Multi-dimensional estimation is not implemented yet."
    assert points.max() <= T, \
        "End time `T` should be larger than the max of `points`."
    
    N = len(points)
    if Delta_n is None: 
        Delta_n_inv = int((N/c)**(1/2))
        Delta_n     = T / Delta_n_inv
    else:
        Delta_n_inv = int(T / Delta_n)
        
    dN, _    = np.histogram(points, bins=Delta_n_inv)
    hat_lamb = dN / Delta_n
    
    if truncation:
        trunc = get_trunc(hat_lamb, Delta_n)
    else:
        trunc = np.repeat(True, len(hat_lamb) - 1)
    
    # Calculate Mean (for better performance, truncation would be helpful.)
    hat_Mean = (hat_lamb[0]*trunc[0] + np.sum(hat_lamb[1:]*trunc)) * Delta_n

    # Calculate BR_const
    hat_Var  = np.concatenate(([hat_lamb[0] if trunc[0] else 0], 
                               hat_lamb[1:][trunc]))
    hat_Var  = np.sum((hat_Var - np.mean(hat_Var))**2)
    hat_BR_const = 1 - np.sqrt(hat_Mean / (Delta_n**2 * hat_Var))
    Del_g_hat    = 1/2 * np.sqrt(hat_Mean / (Delta_n**2 * hat_Var**3))
    BR2_EstAVar  = Del_g_hat**2 * 2 * (hat_Var/T)**2 * T
    
    # Calculate RV(Delta_n)
    diff    = (hat_lamb[1:] - hat_lamb[:-1]) * trunc
    hat_RV1 = np.sum(diff**2)
    hat1    = np.sum(diff**4) / Delta_n
    hat2    = np.sum(hat_lamb[1:] * diff**2) / Delta_n**2
    hat3    = np.sum(hat_lamb[1:]**2 * trunc) / Delta_n**3
    
    # Calculate RV(2Delta_n)
    Delta_n    *= 2
    Delta_n_inv = int(T/Delta_n)
    
    dN, _    = np.histogram(points, bins=Delta_n_inv)
    hat_lamb = dN / Delta_n
    
    if truncation:
        trunc = get_trunc(hat_lamb, Delta_n)
    else:
        trunc = np.repeat(True, len(hat_lamb) - 1)
    
    diff    = (hat_lamb[1:] - hat_lamb[:-1]) * trunc
    hat_RV2 = np.sum(diff**2)
    
    # Estimate IV
    hat_IV = 1 / N**2 * (2*hat_RV2 - hat_RV1/2) 
    
    # Estimate BR
    hat_BR = 1 - np.sqrt(3/2 * hat_Mean / ((Delta_n/2)**2 * (hat_RV1-hat_RV2)))
    
    # Estimate AVar matrix
    hat_eta       = 2/3 * (1/2*Delta_n)**2 * (hat_RV1 - hat_RV2) / hat_Mean
    EstAVar       = np.zeros((3,3))
    EstAVar[0][0] = 2/3 * (hat_RV1 - hat_RV2)
    EstAVar[1][1] = 3/4 * hat1 - (3*hat_eta) * hat2 + (3*hat_eta)**2 * hat3
    EstAVar[1][2] = 29/24 * 3/4 * hat1 \
                    - 23/8 * (3*hat_eta) * hat2 + 21/24 * (3*hat_eta)**2 * hat3
    EstAVar[2][1] = EstAVar[1][2]
    
    hat1 = np.sum(diff**4) / Delta_n
    hat2 = np.sum(hat_lamb[1:] * diff**2) / Delta_n**2
    hat3 = np.sum(hat_lamb[1:]**2 * trunc) / Delta_n**3
    EstAVar[2][2] = 3/2 * hat1 \
                    - 2 * (3*hat_eta) * hat2 \
                    + 2 * (3*hat_eta)**2 * hat3
    
    Delta_n /= 2  # Bring back original Delta_n
    
    # Estimate AVars
    g =  np.array([0,-1/2,2])
    Mean_EstAVar   = EstAVar[0][0]
    IV_EstAVar     = 1/N**4 * g.T @ EstAVar @ g
    RV1_EstAVar    = EstAVar[1][1]
    RV2_EstAVar    = EstAVar[2][2]
    DiffRV_EstAVar = np.array([0,1,-1]).T @ EstAVar @ np.array([0,1,-1])
    
    tmp = 1/2 * np.sqrt(3/2 * hat_Mean / (Delta_n**2 * (hat_RV1 - hat_RV2)**3))
    Del_f_hat   = np.array([0, tmp, -tmp])
    BR_EstAVar  = Del_f_hat.T @ EstAVar @ Del_f_hat
    
    Del_g_hat  = np.array([Del_g_hat, 0, 0])
    EstSigma   = np.array([[2,   4,   1],
                           [4,  12, 3/2], 
                           [1, 3/2, 3/2]]) * (1/2 * hat_RV1)**2
    DiffBR_EstAVar = (Del_f_hat-Del_g_hat).T @ EstSigma @ (Del_f_hat-Del_g_hat)
    
    if recursion:
        if np.isnan(hat_BR):
            return BR_estimation(points, truncation=truncation, T=T, c=.45,
                                 recursion=False)
        # For low BR case, narrower Delta_n would be better.
        if hat_BR < .2:
            return BR_estimation(points, truncation=truncation, T=T, c=.1,
                                 recursion=False)
        if hat_BR < .4:
            return BR_estimation(points, truncation=truncation, T=T, c=.2,
                                 recursion=False)
        # For high BR case, broader Delta_n would be better.
        if hat_BR > .7:
            return BR_estimation(points, truncation=truncation, T=T, c=1,
                                 recursion=False)
        # NOTE: In the simulation, this choice of `c` is known to be the best. 

    out = {'N'   : N,        'Delta_n' : Delta_n,
           'Mean': N,        'SD(Mean)': np.sqrt(Mean_EstAVar)*Delta_n,
           'IV'  : hat_IV,   'SD(IV)'  : np.sqrt(IV_EstAVar*Delta_n),
           'BR'  : hat_BR,   'SD(BR)'  : np.sqrt(BR_EstAVar*Delta_n),
           'BR_const': hat_BR_const, 
           'SD(BR_const)': np.sqrt(BR2_EstAVar*Delta_n),
           'BR - BR_const': hat_BR - hat_BR_const,
           'SD(BR - BR_const)': np.sqrt(DiffBR_EstAVar*Delta_n),
          }
    if debug:
        out.update({'hat_RV2': hat_RV2, 'hat_RV1': hat_RV1,
                          'EstAVar': EstAVar})
        
    return pd.Series(out)


def Hardiman(points, 
             Delta_n=None, T=1, c=1):
    """Hardiman and Bouchaud's Nonparametric Branching Ratio Estimator
    
    This implementation is the nonparametric branching ratio estimator under 
    the constant baseline described in the paper `Branching-ratio approximation
    for the self-exciting  Hawkes process` by Hardiman and Bouchaud (2014, 
    Phys. Rev. E).

    Parameters
    ----------
    points : np.array or list
        A collection of time points of the data. It should be 1-dimensional 
        (univariate).
    Delta_n : float, optional
        A length of the localization interval. If it is `None`, it 
        automatically set to be one over square root of the number of `points`,
        $1/\sqrt{N}$.
    T : float, optional
        End time of the `points`. 
    c : float, optional
        The coefficient for determining the length of interval `Delta_n`.
    """
    points = np.array(points)
    
    if len(points) <= 4:
        return pd.Series({'N': np.nan, 'BR': np.nan})
    
    assert len(points.shape) == 1, \
        "Multi-dimensional estimation is not implemented yet."
    assert points.max() <= T, \
        "End time `T` should be larger than the max of `points`."

    N = len(points)
    if Delta_n is None: 
        Delta_n_inv = int((N/c)**(1/2))
        Delta_n     = T / Delta_n_inv
    else:
        Delta_n_inv = int(T / Delta_n)
        
    dN, _    = np.histogram(points, bins=Delta_n_inv)
    hat_lamb = dN / Delta_n

    hat_Mean = np.sum(hat_lamb) * Delta_n 
    hat_Var  = np.sum((hat_lamb - hat_lamb.mean())**2)

    hat_BR_const = 1 - np.sqrt(hat_Mean / (Delta_n**2 * hat_Var))

    return pd.Series({'N': len(points), 'BR': hat_BR_const})


def AVG_Hardiman(points, divisions):
    if len(points) <= 4:
        return pd.Series({'N': np.nan, 'BR': np.nan})

    BRs = []
    for i in range(divisions):
        chunk = points[(i/divisions <= points) & (points < (i+1) / divisions)]
        if len(chunk) > 4:
            BR = Hardiman(chunk)['BR']
            if not np.isinf(BR):
                BRs.append(BR)

    return pd.Series({'N': len(points), 'BR': np.nanmean(BRs)})


def AVG_Hardiman_4(points):
    return AVG_Hardiman(points, divisions=4)


def AVG_Hardiman_30min(points):
    return AVG_Hardiman(points, divisions=13)


def AVG_Hardiman_5min(points):
    return AVG_Hardiman(points, divisions=78)


# -----------------------------------------------------------------------------
#  tick's Estimation
#
import itertools
from tick.hawkes import HawkesExpKern, HawkesSumExpKern, HawkesEM


def Exp_estimation(points, 
                   verbose=False, penalty='none', tol=1e-05,
                   decay_set=np.logspace(-2, 2, 10)):
    """MLE estimation with exponential kernel with constant baseline"""

    if len(points) <= 4:
        return pd.Series({'N': np.nan, 'BR': np.nan})

    # Since `tick` MLE estimation only works for a fixed decay parameter, we 
    # have to find the best decay parameter.
    best_score = -1e100
    rescaled_points = points * len(points)
    for decay in decay_set:
        hawkes = HawkesExpKern(decays=[[decay]], penalty=penalty, tol=tol)
        hawkes.fit([[rescaled_points]])

        score = hawkes.score()
        if score > best_score:
            if verbose:
                print(' obtained {}\n with {}\n'.format(score, decay))
            best_decays = decay
            best_hawkes = hawkes
            best_score  = score
            
    BR = best_hawkes.adjacency.sum()
    del hawkes
    
    return pd.Series({'N': len(points), 'BR': BR, 'best decay': best_decays})


def SumExp_estimation(points, verbose=False, 
                      decay_set = np.logspace(-2, 2, 10)):
    """MLE estimation with sum of exponential kernel with constant baseline"""

    if len(points) <= 4:
        return pd.Series({'N': np.nan, 'BR': np.nan})

    # Since `tick` MLE estimation only works for a fixed decay parameter, we 
    # have to find the best decay parameter.
    best_score = -1e100
    rescaled_points = points * len(points)
    for decays in itertools.product(decay_set, decay_set):
        hawkes = HawkesSumExpKern(decays=np.array(decays), n_baselines=10,
                                  period_length=len(points))
        hawkes.fit([[rescaled_points]])

        score = hawkes.score()
        if score > best_score:
            if verbose:
                print(' obtained {}\n with {}\n'.format(score, decays))
            best_decays = decays
            best_hawkes = hawkes
            best_score  = score
            
    BR = best_hawkes.adjacency.sum()
    del hawkes
    
    return pd.Series({'N': len(points), 'BR': BR, 'best decay': best_decays})


def EM_estimation(points, support=10, kernel_size=10):
    """EM algorithm with constant baseline and piece-wise constant kernel"""

    if len(points) <= 4:
        return pd.Series({'N': np.nan, 'BR': np.nan})

    rescaled_points = points - points[0]
    rescaled_points = rescaled_points / rescaled_points[-1] * len(points)

    em = HawkesEM(support, kernel_size=kernel_size, n_threads=1, 
                  verbose=False, tol=1e-3)
    em.fit([[rescaled_points]])
    
    BR = em.kernel.sum() * support / kernel_size
    del em
    
    return pd.Series({'N': len(points), 'BR': BR})
#
# -----------------------------------------------------------------------------

