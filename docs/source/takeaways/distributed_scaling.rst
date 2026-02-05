Distributed Scaling Analysis
============================

**Key Message**: Distributed processing significantly improves denoiser performance, especially for larger images. 

High Resolution Color Image
---------------------------

The following dashboard analyzes the scaling performance on a high-resolution color image. It shows the strong scaling (speedup vs number of GPUs), parallel efficiency vs GPUs, and the speedup of gradient and denoiser components.

.. raw:: html

   <iframe src="../_static/images/scaling_highres_color_image/dashboard_scaling.html" width="100%" height="800px" style="border: none; border-radius: 5px;"></iframe>


Tomography 2D
-------------

The same analysis is performed for the 2D Tomography problem.

.. raw:: html

   <iframe src="../_static/images/scaling_tomography_2d/dashboard_scaling.html" width="100%" height="800px" style="border: none; border-radius: 5px;"></iframe>

**Observations**

The distributed implementation of the denoiser performs well, particularly for larger images. In contrast, distributing the physics operator yields limited improvement. This is likely because the physics operator is relatively fast, causing the overhead of distributing it across multiple GPUs to outweigh any potential gains. The denoiser, however, is computationally intensive, making it well suited for distributed execution and resulting in substantial speedups. Overall, this highlights that distributed processing is most effective when applied to true computational bottlenecks.