High-Resolution Color Image Deblurring
==========================================================

This tutorial walks through a complete example of the benchmark using the **High-Resolution Color Image** dataset. We will explore the dataset, configuration, and interpretation of results.


The Dataset: High-Resolution Color Image
-----------------------------------------

**What is it?**

The high-resolution color image dataset is a multi-operator inverse problem where the goal is to **recover a sharp color image from multiple blurred measurements**. Each measurement is the original image degraded by a different anisotropic (directional) Gaussian blur.

**Real-world analogy**: Imagine taking photos of the same scene through different blurred lenses. Each photo is degraded differently. The goal is to recover the original sharp image using information from all these blurred observations.


**Dataset Preview**

.. image:: ../_static/images/main_images/highres_preview.png
   :alt: High-res color dataset preview
   :align: center
   :width: 100%

**Left to right**: Ground truth (original sharp image), Measurement 1 (blur angle 0°), Measurement 2 (blur angle 25.7°)

Configuration: Experiment Setup
-------------------------------

**Benchmark Purpose**

This is a toy example designed to test **distributed computing efficiency** on large-scale inverse problems. The high-resolution image (2048×2048) combined with forward operators (8 blurs) and a pretrained prior (DRUNet) creates computational demands that benefit from distributed processing across multiple GPUs. 

We use the configuration file ``configs/highres_imaging.yml`` to set up the experiment. The configuration specifies:

1. **Objective**: What metric to evaluate (PSNR, etc.)
2. **Dataset**: Which dataset to use and its parameters
3. **Solver**: Which solver to run and its hyperparameters
4. **Execution Grid**: Different GPU configurations to benchmark and compare

Execution Grid
~~~~~~~~~~~~~~

The execution grid specifies different GPU configurations:

.. code-block:: yaml

   slurm_gres, slurm_ntasks_per_node, slurm_nodes, distribute_physics, distribute_denoiser, patch_size, overlap, max_batch_size: [
     ["gpu:1", 1, 1, false, false, 0, 0, 0],
     ["gpu:2", 2, 1, true, true, 448, 32, 0],
     ["gpu:4", 4, 1, true, true, 448, 32, 0],
     ]

**Important**: To ensure proper distributed parallelization, the number of tasks per node (``slurm_ntasks_per_node``) must equal the number of GPUs (``slurm_gres``). This ensures one process per GPU.

This creates a grid comparing three configurations:

**Configuration 1: Single GPU (baseline)**

.. code-block:: yaml

   ["gpu:1", 1, 1, false, false, 0, 0, 0]

- ``slurm_gres: gpu:1`` = 1 GPU
- ``slurm_ntasks_per_node: 1`` = 1 process per GPU
- ``distribute_physics: false`` = No parallelization of blur operators
- ``distribute_denoiser: false`` = Full image processed at once
- **Purpose**: Baseline performance on limited resources


**Configuration 2: 2-GPU with Distribution**

.. code-block:: yaml

   ["gpu:2", 2, 1, true, true, 448, 32, 0]

- ``slurm_gres: gpu:2`` = 2 GPUs
- ``slurm_ntasks_per_node: 2`` = 2 processes (one per GPU)
- ``distribute_physics: true`` = **Split 8 blur operators across 2 GPUs** (4 each)
- ``distribute_denoiser: true`` = **Spatial tiling**: Split large image into patches
- ``patch_size: 448`` = Each patch is 448×448 pixels
- ``overlap: 32`` = Overlap region for smooth transitions between patches
- **Purpose**: Multi-GPU scalability with operator and spatial distribution


**Configuration 3: 4-GPU with Increased Distribution**

.. code-block:: yaml

   ["gpu:4", 4, 1, true, true, 448, 32, 0]

- Similar to Config 2, but with 4 GPUs.
- **Purpose**: Further scalability test with more GPU resources

For more details on distributed computing, see the `DeepInv distributed documentation <https://deepinv.github.io/deepinv/user_guide/reconstruction/distributed.html>`_.

Dataset Parameters
~~~~~~~~~~~~~~~~~~~

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



Interpreting Results
--------------------

**Benchmark Results**

Below is an interactive dashboard comparing the three configurations:

.. raw:: html

   <iframe src="../_static/images/highres_color_image/dashboard.html" width="100%" height="800px" style="border: none; border-radius: 5px;"></iframe>

Interpretation
~~~~~~~~~~~~~~

**PSNR vs. Iteration Count & PSNR vs. Computation Time**

All configurations converge to similar final PSNR values, confirming that distributed processing preserves reconstruction quality. Multi-GPU configurations reach the target PSNR significantly faster than the single-GPU baseline, demonstrating computational efficiency gains.

**Time Breakdown: Gradient and Denoiser**

The stacked bar chart illustrates the computational time spent on gradient computation and denoiser execution for each configuration. The denoiser clearly dominates the overall runtime, making it the most computationally intensive component. Increasing the number of GPUs substantially reduces denoiser execution time, while gradient computation time remains largely unchanged. This demonstrates the effectiveness of distributing the denoiser across multiple GPUs to alleviate the primary performance bottleneck. In contrast, because gradient computation is relatively inexpensive, communication overhead in the distributed setting slightly increases total runtime compared to the single-GPU case.

**GPU Memory Usage: Gradient and Denoiser**

This figure presents GPU memory consumption during gradient computation and denoiser execution. Tracking peak allocated memory is essential for preventing out-of-memory (OOM) errors. For multi-GPU configurations, the reported values correspond to rank 0 (the main process), which is representative of the other ranks.

An interesting behavior is observed in which the 2-GPU configuration exhibits higher denoiser memory usage (4.3 GB) than the single-GPU setup (1.6 GB). This behavior can arise because, for certain denoiser models, peak memory allocation is highly sensitive to image dimensions, even when the effective input size changes only slightly. Further discussion of this effect is provided on the :doc:`takeaways page <../takeaways/denoiser_scaling>`.

**Key Insights**

1. **Consistency**: All configurations achieve similar final PSNR, confirming that distributed processing preserves reconstruction quality.
2. **Efficiency**: Multi-GPU configurations reach the target PSNR faster (lower time-to-convergence).
3. **Bottleneck**: The denoiser is the primary bottleneck in this problem and benefits significantly from distribution across multiple GPUs.

