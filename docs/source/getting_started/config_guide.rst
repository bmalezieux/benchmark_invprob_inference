Configuration Guide
===================

This guide explains how the benchmark configuration works and how to customize it, using ``configs/highres_imaging.yml`` as a concrete example.

Config Files Location
~~~~~~~~~~~~~~~~~~~~~

All configs are in ``configs/`` directory:

- ``highres_imaging.yml`` — Single-process high-res image
- ``tomography_2d.yml`` — 2D tomography
- ``tomography_3d.yml`` — 3D tomography


Anatomy of ``highres_imaging.yml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is the actual structure from ``configs/highres_imaging.yml`` with annotations.

.. code-block:: yaml

   # Select which objective(s) to run. BenchOpt filters available objectives
   # by name, so this runs only the reconstruction objective.
   objective:
     - reconstruction_objective

   # Choose a dataset and its parameters.
   # Here, we use a high‑resolution color image with controllable size and noise.
   dataset:
     - highres_color_image:
         image_size: 2048        # pixels per side (e.g., 2048x2048)
         num_operators: 8        # number of imaging operators to compose
         noise_level: 0.1        # additive noise level

   # Pick solver(s) and their parameters. Here we run Plug‑and‑Play (PnP)
   # with DRUNet as the denoiser.
   solver:
     - PnP:
         denoiser: drunet        # denoiser name
         denoiser_sigma: 0.005   # noise level expected by the denoiser
         step_size: 0.1          # optimization step size
         init_method: ["zeros"]  # initialization policy

         # Grid of execution settings and strategy flags.
         # Each entry defines a combination to run. BenchOpt will run all.
         # Fields:
         #  - slurm_gres: GPU resource request (e.g., "gpu:1")
         #  - slurm_ntasks_per_node: number of tasks per node
         #  - slurm_nodes: number of nodes
         #  - distribute_physics: parallelize physics operator
         #  - distribute_denoiser: parallelize denoiser
         #  - patch_size: image patch size          #  - receptive_field_size: denoiser receptive field size
         #  - max_batch_size: denoiser batch size cap
         slurm_gres, slurm_ntasks_per_node, slurm_nodes, distribute_physics, distribute_denoiser, patch_size, receptive_field_size, max_batch_size: [
           ["gpu:1", 1, 1, false, false, 0,   0,  0],
           ["gpu:2", 2, 1, true,  true, 448, 32, 0],
         ]

   # Global run controls
   max-runs: 10          # how many solver executions are performed to produce the convergence curve
   n-repetitions: 1      # repeat each run for variance estimates
   plot: true            # produce matplotlib plots
   html: true            # generate the HTML summary report




Customizing Your Config
~~~~~~~~~~~~~~~~~~~~~~~

1. Copy an existing config to ``configs/my_config.yml``.
2. Edit dataset parameters (e.g., ``highres_color_image``, ``image_size``, ``noise_level``) and solver hyperparameters (e.g., ``step_size``, ``denoiser_sigma``).
3. Adjust the execution grid to match your hardware 
4. Run the benchmark:

   .. code-block:: bash

      benchopt run . --parallel-config ./configs/config_parallel.yml --config ./configs/my_config.yml




