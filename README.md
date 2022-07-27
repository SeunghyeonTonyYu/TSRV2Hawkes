# TSRV2Hawkes

Source code for [A tale of two time scales: Application in Nonparametric Hawkes processes with Ito semimiartingale Baseline](https://arxiv.org/abs/2102.12783).


## Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [tick](https://x-datainitiative.github.io/tick/) for Simulation.


## Quick Usage
Download the Github repository and execute the following codes at the root directory:
```python
import numpy as np
from main import TSRV2Hawkes

# Generate sample data: Poisson point process
points = np.sort(np.random.uniform(0,1, size=100000))

# Estimation
res = TSRV2Hawkes(points)
print(f"Estimated branching ratio: {res['BR']:.3f}")
print(f"Estimated standard error of branching ratio: {res['SD(BR)']:.3f}")
```

```python
from tick.base import TimeFunction
from tick.hawkes import SimuHawkesExpKernels

# Generate sample data: Hawkes process with branching ratio (BR) = 0.8
n     = 100000
t     = np.linspace(0, 1, num=5*n)
y     = n * np.maximum(np.sin(2 * np.pi * t), 0.2)
base  = np.array([TimeFunction((t, y))])
BR    = np.array([[0.8]])
decay = 2. * n

hawkes = SimuHawkesExpKernels(BR, decay, baseline=base, 
                              end_time=1, verbose=False)
hawkes.simulate()
points = hawkes.timestamps[0]

# Estimation
res = TSRV2Hawkes(points)
print(f"Estimated branching ratio: {res['BR']:.3f}")
print(f"Estimated standard error of branching ratio: {res['SD(BR)']:.3f}")
```


## Full Instructions
1. Put the **data** folder inside the root folder, modify the **data** entry in **run.sh** accordingly. The datasets are available [here](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w&usp=sharing).
1. run `bash run.sh` to compile the Cython files for simulation.


> **Note**
> * To make this estimator valid, the data points should satisfy so called "in-fill" (or heavy-traffic) asymptotics in the sense that the kernel function localized as the number of points increase. Therefore, if the memory kernel has long-memory property, then as Filimonov and Sornette (2015) pointed out, the estimator would suffer from distinguishing long-memory of self-exciting and exogenous time-varying baseline.
> * To have meaningful estimation, we recommend to have at least 5,000 number of points in the data.
>
> **For Simulation**
> * We conducted Monte Carlo simulation with 1,000 repetitions. It takes few hours for non-Power kernel Hawkes processes (depends on the computing machine), and it takes few days for Power kernel Hawkes processes. So we strongly recommend to do simulation without Power-law kernel models.
>
> **For Empirical Study**
> * We have purchased S&P futures E-mini data from [TickMarketData](https://www.tickdatamarket.com/), so you should collect your own tick data to run **Empirics.ipynb**.
>
> **Others**
> * `*.log` files are for tracking processing time. They are not essential for running.
> * All remaining errors are my own.

## Reference

Please cite the following paper if you use this code.

```
@article{yupotiron2022TSRV2Hawkes,
  title={A tale of two time scales: Applications in Nonparametric Hawkes processes with Ito semimartingale Baseline},
  author={Yu, Seunghyeon and Potiron, Yoann},
  journal={Available at SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4174251},
  year={2020}
}
```
