

Benchmark on Inference for Large-Scale Inverse Problems
========================================================

Welcome to the Benchmark on Inference for Large-Scale Inverse Problems — a `BenchOpt <https://benchopt.github.io>`_ benchmark for solving inverse problems in imaging using `DeepInv <https://deepinv.github.io>`_.

Overview
--------

This benchmark evaluates reconstruction algorithms for large-scale inverse problems across multiple imaging modalities. 
In inverse problems, we aim to recover the original signal :math:`x` from measurements :math:`y` following the model:

.. math::

   y = Ax + n

where :math:`A` is the forward operator (e.g., blur, tomographic projection) and :math:`n` represents noise.

**Datasets**

The benchmark includes three imaging scenarios:

- **Tomography (2D/3D):** Computed tomography reconstruction from multiple projection operators
- **High-Resolution Color Images:** Image restoration from multiple anisotropic Gaussian blur operators
- **SKA (Square Kilometre Array):** Radio astronomy imaging from interferometric measurements

These datasets are multi-operator problems: from a single ground truth, we observe different measurements (e.g., tomography uses different projection angles; natural images use different blur kernels). The goal is to recover the original image from these measurements. 
Moreover, the high-resolution images are large-scale, which highlights the importance of scalability.

**Reconstruction Methods**

We leverage the `DeepInv <https://deepinv.github.io>`_ package to implement two approaches:

- **Plug-and-Play (PnP):** Combines data-fidelity optimization with pretrained denoisers as image priors, offering flexibility and strong performance without task-specific training
- **Unrolled Networks:** Deep learning architectures that unfold iterative optimization algorithms

**Evaluation Conditions**

The benchmark assesses solver performance under varying configurations:

- **Image sizes:** Testing across different resolution scales
- **Computational resources:** From single GPU to multi-GPU distributed setups

Performance is measured through reconstruction quality (PSNR) and computational efficiency (runtime, memory usage).

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   examples/index
   takeaways

Quick Links
-----------

- **Get Started:** See :doc:`getting_started/quickstart` for a quick setup guide.
- **Examples:**  Explore :doc:`examples/index` for detailed, dataset-specific examples.
- **Key Takeaways:** Check out :doc:`takeaways` for a summary of benchmark insights and best practices.

