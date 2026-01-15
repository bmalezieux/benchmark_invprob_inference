Configuration Guide
===================

This guide explains how to configure and customize the benchmark. As mentioned in the quickstart, you need two configuration files:

- ``--parallel-config`` — SLURM cluster settings (CPUs, GPUs, walltime, modules)
- ``--config`` — Experiment definition (datasets, solvers, parameters, execution grid)

Available Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All configuration files are located in the ``configs/`` directory:

========================== ============================================
File                       Purpose
========================== ============================================
``config_parallel.yml``    SLURM cluster configuration
``highres_imaging.yml``    High-resolution image reconstruction
``tomography_2d.yml``      2D tomography experiments
``tomography_3d.yml``      3D tomography experiments
========================== ============================================

Parallel Configuration (SLURM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``--parallel-config`` file (e.g., `configs/config_parallel.yml <../../.../../configs/config_parallel.yml>`_) defines cluster-level settings. Here is an example for a SLURM cluster (Jean-Zay):

.. code-block:: yaml

   backend: submitit
   slurm_time: 30                    # Job walltime in minutes
   slurm_additional_parameters:
     cpus-per-task: 10               # CPU cores per job
     qos: qos_gpu-dev                # Queue used for the jobs
     account: your_account_here      # Your cluster account
     constraint: v100-32g            # GPU type (e.g., V100 with 32GB)
   slurm_setup:                      # Commands run before job starts
     - module purge
     - module load pytorch-gpu/py3/2.7.0
     - export NCCL_DEBUG=INFO

**Key parameters to customize:**

- ``cpus-per-task`` — Number of CPU cores per job
- ``slurm_time`` — Maximum runtime (minutes)
- ``constraint`` — GPU type (e.g., ``v100-32g``, ``a100``)
- ``slurm_setup`` — Environment modules and variables for your cluster

**Note:** The number of GPUs per job is specified in the main config file, not here.

Main Configuration (Experiments)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``--config`` file (e.g., `configs/highres_imaging.yml <../../../../configs/highres_imaging.yml>`_) defines what experiments to run. Let's walk through its structure:

**1. Dataset Configuration**

Specify the dataset and its parameters:

.. code-block:: yaml

   dataset:
     - highres_color_image:
         ...


Dataset implementations are in ``datasets/`` (e.g., `datasets/highres_color_image.py <../../../../datasets/highres_color_image.py>`_).

**2. Solver Configuration**

Define which reconstruction algorithms to run:

.. code-block:: yaml

   solver:
     - PnP:
         ...

Available solvers are in ``solvers/`` (e.g., `solvers/pnp.py <../../../../solvers/pnp.py>`_).

**3. Execution Grid**

This section defines GPU configurations and parallelization strategies. Each row specifies one complete experiment configuration:

.. code-block:: yaml

   slurm_gres, slurm_ntasks_per_node, slurm_nodes, distribute_physics, distribute_denoiser, patch_size, receptive_field_size, max_batch_size: [
     ["gpu:1", 1, 1, false, false, 0,   0,  0],
     ["gpu:2", 2, 1, true,  true,  448, 32, 0],
   ]

**Key fields:**

- ``slurm_gres`` — GPU resource request (``"gpu:1"`` = 1 GPU, ``"gpu:2"`` = 2 GPUs)
- ``slurm_ntasks_per_node`` — Number of parallel tasks per node
- ``slurm_nodes`` — Number of compute nodes
- ``distribute_physics`` — Parallelize physics operators across GPUs
- ``distribute_denoiser`` — Enable spatial tiling: split large images into patches and process them in parallel on available GPUs



**Example configurations:**

**Single GPU (no parallelization):**

.. code-block:: yaml

   ["gpu:1", 1, 1, false, false, 0, 0, 0]

Uses 1 GPU with no distributed processing.

**Multi-GPU (2 GPUs in parallel):**

.. code-block:: yaml

   ["gpu:2", 2, 1, true, true, 448, 32, 0]

Uses 2 GPUs in parallel on one node, distributing both physics operators and denoiser. 
For more details on distributed computing, see the `DeepInv distributed documentation <https://deepinv.github.io/deepinv/user_guide/reconstruction/distributed.html>`_.

Customizing Your Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create your own experiment:

1. Copy an existing config:

   .. code-block:: bash

      cp configs/highres_imaging.yml configs/my_experiment.yml

2. Edit dataset parameters (image size, noise levels, etc.)

3. Adjust solver hyperparameters (step size, iterations, denoiser settings)

4. Configure GPU resources in the execution grid

5. Run your benchmark:

   .. code-block:: bash

      python -m benchopt run . \
          --parallel-config ./configs/config_parallel.yml \
          --config ./configs/my_experiment.yml




