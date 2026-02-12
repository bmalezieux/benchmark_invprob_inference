Distributed Scaling Insights
============================

**Key Message**: Distributed processing significantly improves denoiser performance, especially for larger images.

Overview
--------

We evaluate three datasets with varying computational demands to understand how distributed processing affects different components of the reconstruction pipeline. The metrics we track include:

* **Speedup**: The ratio of execution time on a single GPU (or baseline) to execution time on N GPUs
* **Parallel Efficiency**: The percentage of ideal linear scaling achieved, calculated as (Speedup / N_GPUs) × 100%
* **Component Analysis**: Individual performance of gradient computation and denoiser operations

The following dashboards present interactive visualizations showing strong scaling curves, parallel efficiency and component-wise speedup analysis for each dataset.



High-Resolution Color Image
---------------------------

This :doc:`dataset<../examples/highres_example>` features a high-resolution color image reconstruction task where the goal is to recover a sharp image from multiple observations, each blurred by a different anisotropic Gaussian kernel.

.. raw:: html

   <iframe src="../_static/images/scaling_highres_color_image/dashboard_scaling.html" width="100%" height="800px" style="border: none; border-radius: 5px;"></iframe>

**Key Observations:**

* The denoiser component scales well with the distributed framework.
* The physics operator shows minimal benefit from distribution, as it is already computationally efficient and the communication overhead outweighs potential gains
* Image resolution strongly affects distributed processing, with larger images gaining the most from parallelization.

Tomography 2D
-------------

This :doc:`dataset<../examples/tomography_2d>` uses the classic Shepp-Logan phantom to demonstrate 2D tomography reconstruction from projections at multiple angles.

.. raw:: html

   <iframe src="../_static/images/scaling_tomography_2d/dashboard_scaling.html" width="100%" height="800px" style="border: none; border-radius: 5px;"></iframe>

**Key Observations:**

* The denoiser component scales efficiently within the distributed framework, especially for larger images.
* The speedup is higher than in the previous example because the earlier rectangular image required extra patches to cover uneven edges, whereas the square image allows more balanced patch distribution.

Tomography 3D
-------------

This :doc:`dataset<../examples/tomography_3d>` represents the most demanding case, involving 3D volumetric reconstruction too large for a single GPU, so testing begins with at least 2 GPUs.

.. raw:: html

   <iframe src="../_static/images/scaling_tomography_3d/dashboard_scaling.html" width="100%" height="600px" style="border: none; border-radius: 5px;"></iframe>

**Key Observations:**

* Scaling performance is excellent, with a 7.44× speedup from 2 to 16 GPUs and 93% parallel efficiency, demonstrating near-optimal use of additional GPU resources.

.. admonition:: Key Takeaway
    :class: tip
   
    Distributed processing significantly improves denoiser performance, especially for larger images. 

