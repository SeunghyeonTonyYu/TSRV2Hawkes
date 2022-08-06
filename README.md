# TSRV2Hawkes

Source code for [A tale of two time scales: Application in Nonparametric Hawkes processes with Ito semimiartingale Baseline](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4174251).


## Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [tick](https://x-datainitiative.github.io/tick/) for Simulation.


## Quick Usage
Download the Github repository (or copy only the file `main.py`) and execute the following codes at the root directory:
```python
import numpy as np
from main import TSRV2Hawkes

# Generate sample data: Poisson point process
points = np.sort(np.random.uniform(0,1, size=100000))

# Estimation
res = TSRV2Hawkes(points)
res
```

```bash
N                    100000.000000
Delta_n                   0.001000
Mean                 100000.000000
SD(Mean)                331.357813
IV                        0.179400
SD(IV)                    0.800896
BR                        0.045661
SD(BR)                    0.031576
BR_const                  0.030569
SD(BR_const)              0.021677
BR - BR_const             0.015092
SD(BR - BR_const)         0.024797
dtype: float64
```

You can select the value as
```python
res['BR']
```
```
0.179400
```
and calculate test statistic as
```python
res['BR'] / res['SD(BR)']
```
```
49.458679753387635
```

For the Hawkes process, we can use a package `tick` as follows:
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
res
```

```bash
N                    207089.000000
Delta_n                   0.002198
Mean                 207089.000000
SD(Mean)               2050.697646
IV                        4.249904
SD(IV)                    3.864593
BR                        0.778090
SD(BR)                    0.015732
BR_const                  0.937128
SD(BR_const)              0.002084
BR - BR_const            -0.159038
SD(BR - BR_const)         0.011905
dtype: float64
```


## Full Instructions
1. Download the datasets [here](https://drive.google.com/file/d/1AnSaaUtj4C7ZSzt3EvLBTr67_7xq1P9E/view?usp=sharing), and unzip at the **data** folder (so at the root directory, you should possible to access `data/equity/` or `data/E-mini_sample`).
1. run `bash run.sh` to compile the Cython files for the simulation.


> **Note**
> * To make this estimator valid, the data points should satisfy so called "in-fill" (or heavy-traffic) asymptotics in the sense that the kernel function localized as the number of points increase. Therefore, if the memory kernel has long-memory property, then as Filimonov and Sornette (2015) pointed out, the estimator would suffer from distinguishing long-memory of self-exciting and exogenous time-varying baseline.
> * To have meaningful estimation, we recommend to have at least 5,000 number of points in the data.
>
> Others
> * `*.log` files are for tracking processing time. They are not necessary to get the main result.
> * All remaining errors are my own.

> **Warning**
>
> For the Simulation,
> * the Monte Carlo simulation will be run with 1,000 repetitions. It takes few hours for non-Power kernel Hawkes processes (depends on the computing machine), and it takes few days for Power kernel Hawkes processes with 20~30 core CPU. So we strongly recommend to do simulation without Power-law kernel models.
>
> For the Empirical Study,
> * We have purchased S&P futures E-mini data from [TickMarketData](https://www.tickdatamarket.com/), and we only provide sample data `ESH1998` and `ESH2013`. So to run **Empirics.ipynb** or **Exploration.ipynb**, you should buy your own tick data from [TickMarketData](https://www.tickdatamarket.com/).


## Reference

Please cite the following paper if you use this code.

```
@article{yupotiron2022TSRV2Hawkes,
  title={A tale of two time scales: Applications in Nonparametric Hawkes processes with Ito semimartingale Baseline},
  author={Yu, Seunghyeon and Potiron, Yoann},
  journal={Available at SSRN: https://ssrn.com/abstract=4174251},
  year={2020}
}
```
