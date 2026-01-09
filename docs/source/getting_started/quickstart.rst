Quickstart
==========

This guide will help you get started with the benchmark in just a few steps.

Installation
~~~~~~~~~~~~

First, install the required dependencies:

.. code-block:: bash

   pip install deepinv benchopt

Next, clone the project repository:

.. code-block:: bash

   git clone https://github.com/bmalezieux/benchmark_invprob_inference.git
   cd benchmark_invprob_inference

Running Your First Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-GPU on Cluster**

For distributed execution across multiple GPUs on a cluster, use:

.. code-block:: bash

   benchopt run . --parallel-config ./configs/config_parallel.yml --config ./configs/highres_imaging.yml

**Multi-Process CPU with torchrun**

You can also run the benchmark on a single machine with multiple CPU processes using ``torchrun``:

.. code-block:: bash

   benchopt run . --parallel-config ./configs/torchrun_config.yml --config ./configs/highres_imaging_torchrun.yml

Viewing Results
~~~~~~~~~~~~~~~

After the benchmark completes, open the generated HTML report:

.. code-block:: bash

   outputs/benchmark_invprob_inference.html

The report contains runtime comparisons, convergence curves, and detailed solver performance metrics.
