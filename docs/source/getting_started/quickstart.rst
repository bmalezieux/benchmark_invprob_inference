Quickstart
==========

Get the benchmark running in a few simple steps. This guide assumes you have access to a SLURM cluster.

Environment Setup
~~~~~~~~~~~~~~~~~

First, connect to your SLURM cluster. This benchmark is designed to run on high-performance computing (HPC) clusters such as Jean-Zay (IDRIS, France) or similar SLURM-based systems.

**Load required modules** (example for Jean-Zay):

.. code-block:: bash

   module load pytorch-gpu/py3/2.7.0

This loads PyTorch with GPU support and Python 3. Check your cluster's documentation for the equivalent module names.

Installation
~~~~~~~~~~~~

Install the required Python packages (first time only):

.. code-block:: bash

   pip install deepinv benchopt

Clone the benchmark repository:

.. code-block:: bash

   git clone https://github.com/bmalezieux/benchmark_invprob_inference.git
   cd benchmark_invprob_inference

Running the Benchmark
~~~~~~~~~~~~~~~~~~~~~

To run the benchmark, it's very simple—just use this command from the project root:

.. code-block:: bash

   python -m benchopt run . 
       --parallel-config ./configs/config_parallel.yml \
       --config ./configs/highres_imaging.yml

**What each argument does:**

- ``.`` — Run the benchmark in the current directory
- ``--parallel-config`` — SLURM configuration (number of GPUs per job, CPU cores, walltime)
- ``--config`` — Experiment definition (dataset, solvers, image sizes, noise levels, parameters)

See :doc:`config_guide` for details on customizing configurations.

**What happens during execution:**

1. **Configuration parsing** — BenchOpt reads both configs and generates a grid of experiments
2. **Job submission** — Each job executes one complete reconstruction pipeline: a solver (PnP or unrolling) running on a specific dataset and parameter combination
3. **Parallel execution** — Each job can run in parallel on multiple GPUs if specified in the SLURM config
4. **Results collection** — Convergence curves (PSNR), runtime, and memory usage are saved for each job

Viewing Results
~~~~~~~~~~~~~~~

After the benchmark completes, open the HTML report:

.. code-block:: bash

   outputs/benchmark_invprob_inference.html

**The report includes:**

- Runtime comparisons across solvers and configurations
- Convergence curves (PSNR vs iterations)
- Memory and computational resource usage
- Interactive plots for detailed exploration
