Key Takeaways
=============

Main Results
------------

**Quality**

- Distributed processing (multi-GPU) preserves reconstruction quality
- No quality loss from spatial tiling or operator distribution

**Speed**

- Multi-GPU achieves faster time-to-target-PSNR
- Near-linear speedup for distributed workloads (need to verify on larger setups)
- The denoiser is the most time-consuming part in PnP reconstruction 

**Scalability**

- Enables processing of larger images exceeding single-GPU memory
- Spatial tiling reduces per-GPU memory requirements


When to Use Multi-GPU
----------------------

**Use multi-GPU when:**

- Large images (≥2048×2048)
- Many operators (8+)
- Production workloads requiring throughput

**Use single-GPU when:**

- Small images (≤1024×1024)
- Few operators (1-2)
- Prototyping and testing


Configuration Tips
------------------

**Dataset**

- Start with smaller ``image_size`` for testing
- Higher ``noise_level`` = harder problem
- More ``num_operators`` = better reconstruction but slower

**Solver**

- ``denoiser_sigma`` should match noise level

**Troubleshooting**

- Out of memory → reduce ``patch_size`` or enable ``distribute_denoiser``
- Quality issues with tiling → increase ``receptive_field_size``


