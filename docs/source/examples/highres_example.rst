High-Resolution Color Image Deblurring
==========================================================

This tutorial walks through a complete example of the benchmark using the **High-Resolution Color Image** dataset. We will explore the dataset, configuration, and interpretation of results.


The Dataset: High-Resolution Color Image
-----------------------------------------

**What is it?**

The high-resolution color image dataset is a multi-operator inverse problem where the goal is to **recover a sharp color image from multiple blurred measurements**. Each measurement is the original image degraded by a different anisotropic (directional) Gaussian blur.

**Real-world analogy**: Imagine taking photos of the same scene through different blurred lenses. Each photo is degraded differently. The goal is to recover the original sharp image using information from all these blurred observations.


**Dataset Preview**

.. image:: ../_static/images/highres_preview.png
   :alt: High-res color dataset preview
   :align: center
   :width: 100%

**Left to right**: Ground truth (original sharp image), Measurement 1 (blur angle 0°), Measurement 2 (blur angle 25.7°)

Configuration: Experiment Setup
-------------------------------

We use the configuration file ``configs/highres_imaging.yml`` to set up the experiment. The configuration specifies:

1. **Objective**: What metric to evaluate (PSNR, etc.)
2. **Dataset**: Which dataset to use and its parameters
3. **Solver**: Which solver to run and its hyperparameters


Dataset Parameters
~~~~~~~~~~~~~~~~~~~~~~

The dataset section configures the high-resolution color image problem:

.. code-block:: yaml

   dataset:
     - highres_color_image:
         image_size: 2048          
         num_operators: 8           
         noise_level: 0.1        

**Parameter Meanings:**

- ``image_size``: The maximum dimension of the image. ``2048``: Large (tests scalability on GPUs)

- ``num_operators``: Number of different blur operators applied to create measurements. ``8`` : 8 measurements with different blur angles from 0° to 180°

- ``noise_level``: Gaussian noise added to the measurements (in pixel intensity 0-1 range). ``0.1`` : Realistic noise (10% noise)

Solver 
~~~~~~

The solver section configures the Plug-and-Play (PnP) reconstruction algorithm:

.. code-block:: yaml

   solver:
     - PnP:
         denoiser: drunet              
         denoiser_sigma: 0.005        
         step_size: 0.1                
         init_method: ["zeros"]       

**Solver Parameters:**

- ``denoiser``: The pretrained denoiser to use as a prior. 

- ``denoiser_sigma``: Noise level hint passed to the denoiser. Helps the denoiser adapt to the noise level.

- ``step_size``: Gradient descent step size. Controls convergence speed and stability.

- ``init_method``: How to initialize the reconstruction. 

Execution Grid
~~~~~~~~~~~~~~

The execution grid specifies different GPU configurations to benchmark and compare:

.. code-block:: yaml

   slurm_gres, slurm_ntasks_per_node, slurm_nodes, distribute_physics, distribute_denoiser, patch_size, receptive_field_size, max_batch_size: [
     ["gpu:1", 1, 1, false, false, 0, 0, 0],
     ["gpu:2", 2, 1, true, true, 448, 32, 0],]

This creates a grid comparing two configurations:

**Configuration 1: Single GPU (baseline)**

.. code-block:: yaml

   ["gpu:1", 1, 1, false, false, 0, 0, 0]

- ``slurm_gres: gpu:1`` = 1 GPU
- ``slurm_ntasks_per_node: 1`` = Single process
- ``distribute_physics: false`` = No parallelization of blur operators
- ``distribute_denoiser: false`` = Full image processed at once
- **Use case**: Baseline performance, memory-constrained GPUs

**Configuration 2: Multi-GPU with Distribution**

.. code-block:: yaml

   ["gpu:2", 2, 1, true, true, 448, 32, 0]

- ``slurm_gres: gpu:2`` = 2 GPUs
- ``slurm_ntasks_per_node: 2`` = 2 parallel processes
- ``distribute_physics: true`` = **Split 8 blur operators across 2 GPUs** (4 each)
- ``distribute_denoiser: true`` = **Spatial tiling**: Split large image into patches
- ``patch_size: 448`` = Each patch is 448×448 pixels
- ``receptive_field_size: 32`` = Overlap region for smooth transitions between patches
- **Use case**: Scalability test, distributed computing efficiency


Interpreting Results
--------------------

**Benchmark Results**

The benchmark compares two configurations:

- **Configuration 1**: Single GPU (baseline)
- **Configuration 2**: Multi-GPU with distributed physics and denoiser 

**PSNR vs. Iteration Count**

.. image:: ../_static/images/highres_psnr_iteration.png
   :alt: PSNR convergence curves for highres dataset
   :align: center
   :width: 100%

This plot shows reconstruction quality (PSNR in dB) as a function of iteration count. 

**Interpretation:**

- Both configurations converge to similar PSNR values (quality is preserved with distribution)
- Multi-GPU setup does not degrade reconstruction quality despite spatial tiling

**PSNR vs. Computation Time**

.. image:: ../_static/images/highres_psnr_time.png
   :alt: PSNR vs runtime for highres dataset
   :align: center
   :width: 100%

This plot shows reconstruction quality (PSNR) versus actual wall-clock time. It directly compares the efficiency of different configurations.

**Interpretation:**

- Multi-GPU runs reach target PSNR faster than the single-GPU baseline

**Key Insights**

1. **Consistency**: Both configurations achieve similar final PSNR, confirming distributed processing doesn't hurt quality
2. **Efficiency**: Multi-GPU reaches target PSNR faster (lower time-to-convergence)
3. **Scalability**: The multi-GPU setup demonstrates distributed solver viability

