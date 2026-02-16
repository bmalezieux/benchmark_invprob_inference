GPU Performance Under the Hood
==============================

**Key Message**: 

*  Small changes in image size can cause massive memory spikes, 
*  Larger batch don’t always mean faster denoising.

Efficient utilization of GPU resources is critical when solving large-scale inverse problems with deep learning priors. Here, we look at how common denoisers perform and how GPU-specific details affect speed, focusing on two main factors: **image dimensions** and **batch size**. 


Image Shape Effects on GPU Memory Performance
---------------------------------------------

As observed in the :doc:`high-resolution color image example <../examples/highres_example>`, a multi-GPU setup can sometimes result in unexpectedly high memory usage (e.g., 4.3 GB vs 1.6 GB) compared to single GPU usage. To understand this phenomenon, we conducted an experiment studying the impact of input image shape on memory allocation.

**Experiment Protocol**:

*   **Models**: DRUNet, UNet, and DnCNN.
*   **Setup**: Fixed image height at **2048 pixels**, varying width from **1024 to 2048 pixels**.
*   **Variation**: We tested width increments of 64 pixels versus irregular increments (e.g., 68 pixels).

Why does this matter? Deep neural networks often rely on operations (like downsampling) that require input dimensions to be divisible by specific factors (e.g., 8, 16, or 32).

.. raw:: html

   <iframe src="../_static/images/takeaway/image_size_vs_memory.html" width="100%" height="480px" style="border: none; border-radius: 5px;"></iframe>

**Observations**

*   **Step 64 (Divisible inputs)**: Memory scaling is linear and predictable.
*   **Step 68 (Non-divisible inputs)**: For **DRUNet**, we observe massive spikes in memory usage for relatively small changes in input size. **Example**: At a width of **1228px**, memory usage is modest (~1.3 GB). Increasing the width slightly to **1296px** causes usage to triple to **~4 GB**.

**Explanation**

1.  **cuDNN Heuristics**: NVIDIA's cuDNN library runs internal benchmarks to select the fastest convolution algorithm for a specific input shape. Irregular shapes can lead to algorithms that are less memory-intensive,causing noticeable drops in memory usage.
2.  **Architecture Depth**: Deep architectures like DRUNet use residual and skip connections that store intermediate activations, this can lead to unexpected changes in memory usage.

Returning to the :doc:`high-resolution color image example <../examples/highres_example>`, the image size is 2048×1366 (3:2 aspect ratio), which is not divisible by 8. This likely explains the difference in memory usage observed in the multi-GPU setup compared to a single GPU.


.. admonition:: Key Takeaway
    :class: tip
    
    Small changes in image dimensions can trigger significant memory spikes. To ensure stable performance, resize or crop inputs so that dimensions are divisible by powers of 2 (e.g., 8, 16, 32).

Batch Size Effects on Runtime
-----------------------------

When using multi-GPU setups for denoiser networks, we typically split images into patches and process them in batches. This raises an important question: **how does batch size impact runtime performance?**


**Experiment Protocol**:

*   **Input**: One high-resolution image (2048x2048).
*   **Strategy**: Image divided into patches of size 128x128, 256x256, or 512x512.
*   **Variable**: Batch size (number of patches processed in parallel): 1, 2, 4, 8, 16.
*   **Metrics**: Peak memory usage and throughput (patches per second and images per second).

.. raw:: html

   <iframe src="../_static/images/takeaway/batch_size_vs_memory.html" width="100%" height="480px" style="border: none; border-radius: 5px;"></iframe>

.. raw:: html

   <iframe src="../_static/images/takeaway/batch_size_vs_throughput.html" width="100%" height="480px" style="border: none; border-radius: 5px;"></iframe>

**Observations**

*   **Memory Usage**: As expected, increasing the batch size increases memory usage.
*   **Small Patches (128x128)**: Throughput improves significantly with larger batch sizes. Small batches underutilize the GPU's parallelism.
*   **Large Patches (256x256, 512x512)**: Throughput quickly saturates. Beyond a certain batch size, increasing the batch does not yield meaningful speed gains, even if GPU memory is available.
*    **Optimal Case**: The highest throughput (images/s) is achieved with a batch size of 16 for 128×128 patches. Exact values may vary depending on the denoiser model and available computational resources.


**Explanation: Why GPU Memory ≠ Speed**

Filling GPU memory does not automatically translate to faster execution:

*   **Memory Bandwidth**: Large tensors require moving data between global memory and compute cores. Even if memory is available, the rate at which data can be transferred (bandwidth) may become the bottleneck. Large batches can saturate memory bandwidth before the GPU’s compute units are fully utilized.
*   **Cache Thrashing**: The GPU's L2 cache is limited. If the working set for a large batch exceeds the cache size, the processor must frequently fetch data from slower global memory, causing stalls that reduce overall throughput.

.. admonition:: Key Takeaway
    :class: tip
   
    Larger batch don’t always mean faster denoising



