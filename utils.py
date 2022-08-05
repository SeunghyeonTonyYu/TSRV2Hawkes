import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import gc
import joblib
import numpy as np
import pandas as pd
from copy import copy
from time import time
from functools import partial
from multiprocessing import Pool, cpu_count
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['seaborn-paper'])
plt.rcParams['figure.dpi'] = 120
import warnings
warnings.filterwarnings('ignore')
from logging import basicConfig, debug, info, warning, error, DEBUG, INFO


IMG_PATH = 'figs/'


from IPython.display import Audio

def alarm():
    """ makes sound on client using javascript (works with remote server) """      
    framerate = 44100
    duration  = .5
    freq      = 300
    t    = np.linspace(0, duration, int(framerate * duration))
    data = np.sin(2 * np.pi * freq * t)

    return Audio(data, rate=framerate, autoplay=True)


def dropna(x):
    x = np.array(x)
    return x[~np.isnan(x)]


def format_minute(start, end, division, to_str=True):
    times = pd.to_timedelta(np.arange(start, end, (end-start)/division),
                            unit='s')
    times = (pd.to_datetime(0) + times)
    if to_str:
        return times.strftime("%H:%M")
    return times


MONTH_CODE = ['H','M','U','Z']
def get_futures_code(time):
    year  = int((time - .25)//1)
    month = MONTH_CODE[int((time%1) / 0.25) - 1]
    
    return month + str(year)

        
def to_0_1(times):
    if len(times) == 0:
        return np.array([])
    elif len(times) == 1:
        return np.array([1])
        
    points  = np.sort(times)
    points -= points.min()
    points /= points.max()
    
    return points


def intersect(*d):
    sets = iter(map(set, d))
    res  = next(sets)
    for s in sets:
        res = res.intersection(s)
        
    return np.array(list(res))