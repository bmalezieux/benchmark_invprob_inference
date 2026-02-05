Convolution Optimization
========================

**Key Message**: When applying large blur effects ("kernels") to images, switching from "standard" convolution to **FFT-based convolution** can make your code run hundreds of times faster (e.g., reducing time from 30s to <0.1s).

What is Convolution?
--------------------

In image processing, **convolution** is the mathematical operation used to apply effects like blurring or sharpening.

Imagine a small window (called the **kernel** or **filter**) sliding over every pixel of your image to calculate a new pixel value.

*   **Small Kernel (e.g., 3x3)**: For every pixel in the image, the computer looks at just its 8 immediate neighbors. This is fast.
*   **Large Kernel (e.g., 20x20)**: To simulate a wide blur (like an out-of-focus camera), the window must look at hundreds of neighbors for *every single pixel* in the image.

The Problem with Standard Convolution
-------------------------------------

Standard convolution (spatial domain) works exactly like that sliding window. As the kernel size increases, the number of calculations required for each pixel grows, significantly slowing down the process.

In our benchmark, using standard convolution for a 20x10 blur kernel on the high-resolution image took **30 seconds per gradient step**.

The FFT Solution
----------------

There is a mathematical shortcut using the **Fast Fourier Transform (FFT)**.

Instead of sliding a window pixel-by-pixel, FFT works like translating the image into a different language (frequency domain) where "blurring" becomes a simple multiplication.

1.  **Translate (FFT)**: Convert the image and blur kernel to frequency domain.
2.  **Multiply**: Simple multiplication (very fast, independent of kernel size).
3.  **Translate Back (Inverse FFT)**: Convert back to a normal image.

**Why is it faster?**
While the "translation" (FFT) has a fixed cost, the multiplication is instant. For large kernels, this fixed cost is tiny compared to the billions of operations required by the sliding window.

DeepInv Implementation
----------------------

In the **DeepInv** library, you can choose between these methods:

1.  **Standard Convolution** (`conv2d/conv_transpose2d`): Good for tiny kernels (e.g., 3x3).
2.  **FFT Convolution** (`conv2d_fft/conv_transpose2d_fft`): **Must be used** for larger kernels.

Performance Comparison
----------------------

In the **High-Resolution usage example**:

*   **Image**: 2048×2048 pixels
*   **Blur Kernel**: ~20×10 pixels

.. list-table::
   :widths: 40 40 20
   :header-rows: 1

   * - Method
     - Time for Gradient Step
     - Status
   * - **Standard (Spatial)**
     - 30.0 seconds
     - Too Slow
   * - **FFT-based**
     - **< 0.1 seconds**
     - Efficient

**Recommendation**

If you are simulating physics like **motion blur**, **defocus**, or **atmospheric turbulence**, the kernel is usually large. Always verify you are using the FFT-based implementation to avoid huge performance penalties.
