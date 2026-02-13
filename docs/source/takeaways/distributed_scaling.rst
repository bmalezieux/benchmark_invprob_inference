Distributed Scaling Insights
============================

**Key Message**: Distributed processing significantly improves denoiser performance, especially for larger images.

Overview
--------

We evaluate three datasets with increasing image/volume sizes and computational demands to understand how distributed processing affects different components of the reconstruction pipeline.

The metrics we track include:

* **Speedup**: The ratio of execution time on a single GPU (or baseline) to execution time on N GPUs
* **Parallel Efficiency**: The percentage of ideal linear scaling achieved, calculated as (Speedup / N_GPUs) × 100%
* **Component Analysis**: Individual performance of gradient computation and denoiser operations

The following dashboards present interactive visualizations showing strong scaling curves, parallel efficiency and component-wise speedup analysis for each dataset.



High-Resolution Color Image
---------------------------

This :doc:`dataset<../examples/highres_example>` features a high-resolution color image reconstruction task where the goal is to recover a sharp image from multiple observations, each blurred by a different anisotropic Gaussian kernel.

In this dataset, we vary the image size from 1024 to 4096 (the largest size that fits in memory) and the number of GPUs from 1 to 16. Some clusters limit GPUs per node (e.g., 4 GPUs on Jean-Zay), so higher GPU counts use multiple nodes. For example, 16 GPUs correspond to 4 nodes with 4 GPUs each:


.. code-block:: yaml

   slurm_gres, slurm_ntasks_per_node, slurm_nodes: 
   [["gpu:4", 4, 4]]

.. raw:: html

   <iframe src="../_static/images/scaling_highres_color_image/dashboard_scaling.html" width="100%" height="800px" style="border: none; border-radius: 5px;"></iframe>

**Key Observations:**

* The denoiser component scales well with the distributed framework.
* The physics operator shows minimal benefit from distribution, as it is already computationally efficient and the communication overhead outweighs potential gains.
* Image resolution strongly affects distributed processing, with larger images gaining the most from parallelization. For the denoiser at 16 GPUs, size 4096 achieves 5.75× speedup versus 1.48× at size 1024.

Tomography 2D
-------------

This :doc:`dataset<../examples/tomography_2d>` uses the classic Shepp-Logan phantom to demonstrate 2D tomography reconstruction from projections at multiple angles.

Similar to the previous example, we vary the image size from 1024 to 2048 (the largest size that fits in memory) and the number of GPUs from 1 to 16.

.. raw:: html

   <iframe src="../_static/images/scaling_tomography_2d/dashboard_scaling.html" width="100%" height="800px" style="border: none; border-radius: 5px;"></iframe>

**Key Observations:**

* The denoiser component scales efficiently within the distributed framework, especially for larger images. For 16 GPUs, size 2048 achieves 6.86× speedup versus 3.64× at size 1024.
* The speedup is higher than in the previous example because the earlier rectangular image required extra patches to cover uneven edges, whereas the square image allows more balanced patch distribution.

Tomography 3D
-------------

This :doc:`dataset<../examples/tomography_3d>` represents the most demanding case, involving 3D volumetric reconstruction too large for a single GPU, so testing begins with at least 2 GPUs. For this dataset, we fix the image size at 512 and vary the number of GPUs from 2 to 16.

.. raw:: html

   <iframe src="../_static/images/scaling_tomography_3d/dashboard_scaling.html" width="100%" height="600px" style="border: none; border-radius: 5px;"></iframe>

**Key Observations:**

* As noted in :doc:`Tomography 3D<../examples/tomography_3d>`, the denoiser dominates runtime (>99%), so the pipeline speedup closely matches the denoiser speedup.
* Scaling performance is excellent, with a 7.44× speedup from 2 to 16 GPUs and 93% parallel efficiency, demonstrating near-optimal use of additional GPU resources.

.. admonition:: Key Takeaway
    :class: tip
   
   Distributed processing significantly improves denoiser performance, especially for larger images.

