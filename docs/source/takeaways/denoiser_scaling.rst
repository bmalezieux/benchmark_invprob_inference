Denoiser Performance
====================

**Key Message**: The performance of deep learning denoisers is non-linear. Small changes in image size can cause massive memory spikes (the "cliff effect"). While reducing the batch size can lower memory usage, it does not always significantly degrade processing speed.

Image Shape Effects on GPU Memory Usage
---------------------------------------

The following interactive dashboard compares 3 common denoisers: **DRUNet**, **GS-DRUNet**, and **DnCNN**.
We fix the image height at **2048 pixels** and vary the width from **1024 to 2048 pixel**.

.. raw:: html

   <iframe src="../_static/images/takeaway/image_scaling_denoiser.html" width="100%" height="480px" style="border: none; border-radius: 5px;"></iframe>

**Observations**

**Computational Time**

*   **Non-Monotonic Behavior**: You might expect processing time to increase strictly as the image gets larger. However, the results show slight fluctuations.
*   **Reason**: Deep neural networks often require input dimensions to be divisible by specific factors (e.g., 8, 16, or 32). When this is not the case, padding and internal rounding can change tensor shapes and trigger different computational paths or kernel selections, resulting in non-linear performance behavior.

**GPU Memory Consumption**

*   **Sudden Jumps**: For **DRUNet**, we observe a massive spike in memory usage for relatively small changes in input size. Example: At a width around **1280px**, memory usage is modest (~1.3 GB). Increasing the width slightly to **1296px** causes usage to triple to **~4 GB**.
*   **Implication**: A configuration that works fine for a 2048x1280 patch might crash with an Out-Of-Memory (OOM) error if the patch size is increased by just a few pixels. This highlights the importance of testing specific patch sizes rather than assuming linear scaling.

Batch Size Performance
----------------------

Increasing the batch size allows the GPU to process multiple inputs in parallel. The following charts compare execution time and memory usage for different batch sizes on **256x256** and **512x512** image patches.

**256x256 Patches**

.. raw:: html

   <iframe src="../_static/images/takeaway/batch_size_denoiser_benchmark_256x256.html" width="100%" height="480px" style="border: none; border-radius: 5px;"></iframe>

**512x512 Patches**

.. raw:: html

   <iframe src="../_static/images/takeaway/batch_size_denoiser_benchmark_512x512.html" width="100%" height="480px" style="border: none; border-radius: 5px;"></iframe>

**Key Insight: Latency vs Throughput**

*   **Memory**: Batch size has a direct impact on memory consumption, as larger batches require storing more input data and intermediate activations. Reducing the batch size is an effective way to lower memory usage and avoid out-of-memory issues.
*   **Execution Time**: As long as sufficient GPU memory is available, increasing the batch size often results in **no significant increase in execution time** and may even reduce it. 

