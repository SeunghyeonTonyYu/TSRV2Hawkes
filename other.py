import numpy as np
from scipy.ndimage import convolve


def PRVEstimator(Y, 
                 K=None, mode='optimal', axis=None):
    """Pre-averaged Realized Volatility Estimator
    
    Parameters
    ----------
    Y : np.array
        Observed log-price array.
    K : int
        Pre-averaging window size.
    mode : {'optimal', 'positive'}
        'optimal':
          By default, mode is 'optimal'.  This returns optimal pre-averaged 
          realized volatility with bias-correction. However, this may return
          negative value.
          
        'positive':
          Mode 'positive' returns pre-averaged realized volatility without 
          bias-correction. It always return positive value.
        
    References
    ----------    
    .. [1] p.240 of Ait-Sahalia and Jacod (2014)
    """
    @np.vectorize
    def kernel(t):
        return 2 * np.min(np.array([t, 1 - t]))
    
    if len(Y.shape) > 1 and axis is None:
        axis = 0
    if axis is None:
        dY = np.diff(Y)
        n  = len(dY)
    else:
        dY = np.diff(Y, axis=axis)
        n  = dY.shape[axis]
    if K is None:
        K  = int(np.round(np.sqrt(n)))
        
    i_K        = np.arange(1,K,1)/K
    kernel_i_K = kernel(i_K)
    phi_K      = 1/K * np.sum(kernel(i_K)**2)
    phi        = 1/3
    
    if axis is None:
        d_barY_sum = np.sum(np.convolve(dY, kernel_i_K, 'valid')**2)
    else:
        half_K = int(len(kernel_i_K)/2)
        remain = (len(kernel_i_K)+1)%2 
        kernel_shape = [1 for _ in range(len(dY.shape))]
        kernel_shape[axis] = -1
        tmp = convolve(dY, kernel_i_K.reshape(*kernel_shape))
        d_barY_sum = np.sum(tmp.take(
            indices=range(half_K, tmp.shape[axis] - half_K - remain), axis=axis)**2,
                         axis)
    
    if mode != 'optimal':
        return 1/phi/K * d_barY_sum
    else:
        d_kernel2 = np.diff(kernel(np.concatenate([[0],i_K,[1]])))**2
        if axis is None:
            bar_dY2_sum = np.sum(np.convolve(dY**2, d_kernel2, 'valid'))
        else:
            half_K = int((len(d_kernel2)-1)/2)
            remain = (len(d_kernel2)+1)%2 
            tmp    = convolve(dY**2, d_kernel2.reshape(*kernel_shape))
            bar_dY2_sum = np.sum(tmp.take(
                indices=range(half_K, tmp.shape[axis] - half_K-remain), axis=axis),
                                 axis)
            
        return 1/phi_K/K/(1-K/n) * (d_barY_sum - 0.5 * bar_dY2_sum)
    
    
def RVEstimator(Y, axis=None):
    """Realized Volatility Estimator"""
    
    if axis is None:
        axis = 0
    dY = np.diff(Y, axis=axis)
    
    return np.sum(dY**2, axis=axis)