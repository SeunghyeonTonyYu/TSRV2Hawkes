import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='TSRV2Hawkes',
    ext_modules=cythonize("_DGP.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)