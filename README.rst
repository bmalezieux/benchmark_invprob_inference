Benchmarking inference for large scale inverse problems 
=====================================================
|Build Status| |Python 3.10+|

This benchmark is dedicated to solvers of inverse problems in large scale settings.

The benchmark compares the performance of different reconstruction algorithms (solvers) on various inverse problems (datasets) such as tomography, deblurring, etc.
It uses standard metrics like PSNR to evaluate the reconstruction quality.

Install
--------

This benchmark requires to be installed using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/bmalezieux/benchmark_invprob_inference
   $ cd benchmark_invprob_inference
   $ pip install -e .

Run the benchmark
-----------------

To run the benchmark, use the ``benchopt run`` command from the root of the repository:

.. code-block::

   $ benchopt run .

You can also specify which solvers and datasets to run:

.. code-block::

   $ benchopt run . -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/bmalezieux/benchmark_invprob_inference/actions/workflows/main.yml/badge.svg
   :target: https://github.com/bmalezieux/benchmark_invprob_inference/actions
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/
