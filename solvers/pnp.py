import os
from datetime import datetime
import time

import torch
from benchopt import BaseSolver
from deepinv.distributed import DistributedContext, distribute
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.physics import Physics, stack
from deepinv.utils.tensorlist import TensorList
from deepinv.loss.metric import PSNR

from benchmark_utils import create_drunet_denoiser
from benchmark_utils.gpu_metrics import GPUMetricsTracker, save_result_per_rank
from benchmark_utils.support_dataloader import _move_measurement_to_device


def compute_step_size_from_operator(
    operator: Physics,
    ground_truth: torch.Tensor,
) -> float:
    """
    Compute step size from Lipschitz constant of operator.

    Args:
        operator: Physics operator (can be stacked or distributed)
        ground_truth: Ground truth tensor (used for creating example signal)
        ctx: Optional distributed context for synchronization

    Returns:
        Step size (1 / lipschitz_constant)
    """
    with torch.no_grad():
        # Create example signal for norm computation
        x_example = torch.zeros_like(
            ground_truth, device=ground_truth.device, dtype=ground_truth.dtype
        )

        # Compute Lipschitz constant
        lipschitz_constant = operator.compute_norm(x_example, local_only=False)

        return 1.0 / lipschitz_constant if lipschitz_constant > 0 else 1.0


def initialize_reconstruction(
    signal_shape: tuple,
    operator: Physics,
    measurements,
    device: torch.device,
    method: str = "pseudo_inverse",
) -> torch.Tensor:
    """
    Initialize reconstruction signal.

    For pseudo-inverse initialization:
        x_0 = A^dagger y

    where A^dagger is the adjoint/pseudo-inverse of the operator.
    For stacked operators, this is handled automatically.

    Args:
        signal_shape: Shape of the signal to initialize
        operator: Physics operator (can be stacked or distributed)
        measurements: Measurements (TensorList for stacked physics)
        device: Device to create tensor on
        method: Initialization method ('zeros' or 'pseudo_inverse')

    Returns:
        Initialized reconstruction tensor
    """
    if method == "zeros":
        return torch.zeros(signal_shape, device=device)

    elif method == "pseudo_inverse":
        x_init = operator.A_dagger(measurements).clamp(0, None)
        return x_init

    else:
        raise ValueError(
            f"Unknown initialization method: {method}. Use 'zeros' or 'pseudo_inverse'"
        )


class Solver(BaseSolver):
    """Plug-and-Play (PnP) solver with optional distributed support."""

    name = "PnP"

    requirements = [
        "pip::torch",
        "numpy",
        "pip::git+https://github.com/deepinv/deepinv.git@main",
    ]

    # Use callback sampling strategy for transparent iteration control
    sampling_strategy = "callback"

    # Solver parameters
    parameters = {
        "denoiser": ["drunet"],
        "denoiser_lambda_relaxation": [None],
        "step_size": [None],
        "step_size_scale": [0.99],
        "denoiser_sigma": [0.05],
        "n_iter": [10],
        "distribute_physics": [False],
        "distribute_denoiser": [False],
        "patch_size": [128],
        "overlap": [32],
        "max_batch_size": [0],
        "init_method": ["pseudo_inverse"],
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
        "torchrun_nproc_per_node": [1],
        "name_prefix": ["pnp"],
    }

    def set_objective(
        self,
        dataloader,
        physics,
        ground_truth_shape,
        num_operators,
        min_pixel=0.0,
        max_pixel=1.0,
    ):
        """Set the objective from the dataset.

        Args:
            dataloader: PyTorch DataLoader containing ground truth and measurements
            physics: Forward operator (stacked physics or list or callable)
            ground_truth_shape: Shape of the ground truth tensor
            num_operators: Number of operators in the physics
        """
        self.dataloader = dataloader
        self.physics = physics
        self.ground_truth_shape = ground_truth_shape
        self.num_operators = num_operators
        self.clip_range = (min_pixel, max_pixel)

        self.world_size = 1
        self.ctx = None

        # Check if distributed environment is already set up
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Already initialized by dataset or previous call
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            print(
                f"Distributed environment already initialized: world_size={self.world_size}"
            )
        else:
            # Try to initialize
            try:
                import submitit

                submitit.helpers.TorchDistributedEnvironment().export(
                    set_cuda_visible_devices=False
                )
                self.world_size = int(os.environ.get("WORLD_SIZE", 1))
                print(
                    f"Initialized distributed environment via submitit: world_size={self.world_size}"
                )
            except ImportError:
                print("submitit not installed, running in non-distributed mode")
            except RuntimeError as e:
                # This could be SLURM not available or other runtime issues
                error_msg = str(e).lower()
                if "slurm" in error_msg or "environment" in error_msg:
                    print(f"SLURM environment not available: {e}")
                else:
                    print(
                        f"RuntimeError initializing submitit (possibly already called): {e}"
                    )
                print("Running in non-distributed mode")

        self.distributed_mode = self.world_size > 1
        self.reconstruction = torch.zeros(
            self.ground_truth_shape
        )  # Will be overwritten per batch
        self.reconstructions = []  # List to store all batch reconstructions

        # Initialize GPU metrics tracker (device will be set in _run_with_context)
        self.gpu_tracker = None

        # Initialize results storage for per-iteration tracking
        self.all_results = []

        # Generate name based on whether using slurm or torchrun
        if hasattr(self, "slurm_ntasks_per_node") and self.slurm_ntasks_per_node > 1:
            self.name = (
                self.name_prefix
                + datetime.now().strftime("_%Y%m%d_%H%M%S_")
                + f"{self.slurm_nodes}n{self.slurm_ntasks_per_node}t"
            )
        elif (
            hasattr(self, "torchrun_nproc_per_node")
            and self.torchrun_nproc_per_node > 1
        ):
            self.name = (
                self.name_prefix
                + datetime.now().strftime("_%Y%m%d_%H%M%S_")
                + f"torchrun_{self.torchrun_nproc_per_node}proc"
            )
        else:
            self.name = (
                self.name_prefix
                + datetime.now().strftime("_%Y%m%d_%H%M%S_")
                + "_single"
            )

        # Add rank to name in distributed mode
        if self.distributed_mode:
            self.name = self.name + f"_rank{int(os.environ.get('RANK', 0))}"

    def run(self, cb):
        """Run the PnP algorithm with callback for iteration control.

        Args:
            cb: Callback function to call at each iteration. Returns False when to stop.
        """
        if self.distributed_mode:
            # Use cleanup=True to properly destroy process group when done
            # This will reuse the process group created by dataset (if any)
            with DistributedContext(seed=42, cleanup=True) as ctx:
                self.ctx = ctx
                self._run_with_context(cb, ctx)
        else:
            self._run_with_context(cb, ctx=None)

    def _setup_components(self, device, ctx=None):
        """Setup denoiser, prior, and data fidelity.

        Args:
            device: Device to use
            ctx: Optional distributed context for distributing components

        Returns:
            Tuple of (prior, data_fidelity)
        """
        if self.denoiser == "drunet":
            # Use the utility function to create appropriate DRUNet
            denoiser = create_drunet_denoiser(
                ground_truth_shape=self.ground_truth_shape,
                device=device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unknown denoiser: {self.denoiser}")

        # Distribute denoiser if context provided and requested
        if ctx is not None and self.distribute_denoiser:
            denoiser = distribute(
                denoiser,
                ctx,
                patch_size=self.patch_size,
                overlap=self.overlap,
                tiling_dims=(
                    (-3, -2, -1) if len(self.ground_truth_shape) == 5 else (-2, -1)
                ),
                max_batch_size=self.max_batch_size,
            )

        # Create prior and data fidelity
        prior = PnP(denoiser=denoiser)
        data_fidelity = L2()

        if ctx is not None and self.distribute_physics:
            data_fidelity = distribute(
                data_fidelity,
                ctx,
            )

        return prior, data_fidelity

    def _compute_step_size(self, physics, device):
        """Compute step size from physics operator.

        Args:
            physics: Physics operator (can be stacked or distributed)
            device: Device to use

        Returns:
            Step size value
        """

        if isinstance(self.step_size, float):
            return self.step_size

        # Create example signal for norm computation
        ground_truth_example = torch.zeros(self.ground_truth_shape, device=device)

        raw_step_size = compute_step_size_from_operator(
            physics,
            ground_truth_example,
        )

        step_size = raw_step_size * self.step_size_scale

        return step_size

    def _initialize_reconstruction(self, physics, measurements, ground_truth, device):
        """Initialize reconstruction signal.

        Args:
            physics: Physics operator (can be stacked or distributed)
            measurements: Measurements
            ground_truth: Ground truth tensor (used to determine signal shape)
            device: Device to use

        Returns:
            Initialized reconstruction tensor
        """
        with torch.no_grad():
            return initialize_reconstruction(
                signal_shape=ground_truth.shape,
                operator=physics,
                measurements=measurements,
                device=device,
                method=self.init_method,
            )

    def _run_pnp_iterations(
        self, prior, data_fidelity, physics, step_size, cb,ctx
    ):
        """Run PnP iterations over all batches.

        Args:
            prior: PnP prior
            data_fidelity: L2 data fidelity
            physics: Physics operators
            step_size: Step size for gradient descent
            cb: Callback function

        Returns:
            Final average PSNR across all batches
        """
        with torch.no_grad():

            while True:
                keep_going = cb()

                if self.distributed_mode and self.ctx is not None:
                    decision = torch.tensor([float(keep_going)], device=self.device)
                    self.ctx.broadcast(decision, src=0)
                    keep_going = bool(decision.item())

                if not keep_going:
                    break
                
                # Init metrics (computed once per image/batch)
                init_psnr_per_image = []  # [num_images]
                init_time_per_batch = []  # [num_batches] 
                
                # Iteration metrics - structure as [num_images][n_iter]
                psnr_per_image_iter = []  # List where each element is a list of PSNR values for one image
                time_per_batch_iter = []  # [num_batches][n_iter]
                
                # GPU metrics per [batch][n_iter]
                all_batch_iter_metrics = {}
                
                # Track total number of images processed to build per-image PSNR lists
                num_images_processed = 0
                
                # Process all batches
                for batch_idx, (ground_truth, measurement) in enumerate(self.dataloader):
                    batch_size = ground_truth.shape[0]
                    
                    # === INITIALIZATION PHASE ===
                    init_start = time.perf_counter()
                    
                    # Move data to device
                    ground_truth = ground_truth.to(self.device)
                    measurement = _move_measurement_to_device(measurement, self.device)

                    # Initialize reconstruction
                    reconstruction = self._initialize_reconstruction(self.physics, measurement, ground_truth, self.device)
                    if self.clip_range is not None:
                        reconstruction = torch.clamp(reconstruction, self.clip_range[0], self.clip_range[1])

                    # Compute and store initial PSNR for each image in batch
                    initial_psnr_values = self.psnr_metric(reconstruction, ground_truth)  # [batch_size]
                    init_psnr_per_image.extend(initial_psnr_values.cpu().tolist())
                    
                    # Record init time for this batch
                    init_time_per_batch.append(time.perf_counter() - init_start)
                    
                    # Initialize per-image PSNR lists for images in this batch
                    for _ in range(batch_size):
                        psnr_per_image_iter.append([])
                    
                    batch_time_list = []  # Will collect iteration times for this batch
                    
                    # === ITERATION PHASE ===
                    for iter_idx in range(self.n_iter):
                        iter_start = time.perf_counter()
                        
                        # Gradient step
                        with self.gpu_tracker.track_step("gradient"):
                            grad = data_fidelity.grad(reconstruction, measurement, physics)
                            reconstruction = reconstruction - step_size * grad

                        # Denoising step
                        with self.gpu_tracker.track_step("denoise"):
                            if self.denoiser_lambda_relaxation is None:
                                reconstruction = prior.prox(reconstruction, sigma_denoiser=self.denoiser_sigma)
                            else:
                                denoised = prior.prox(reconstruction, sigma_denoiser=self.denoiser_sigma)
                                lamda = self.denoiser_lambda_relaxation
                                alpha = (step_size * lamda) / (1 + step_size * lamda)
                                reconstruction = (1 - alpha) * reconstruction + alpha * denoised

                            if self.clip_range is not None:
                                reconstruction = torch.clamp(reconstruction, self.clip_range[0], self.clip_range[1])

                        # Compute PSNR for each image and store in per-image lists
                        iter_psnr_values = self.psnr_metric(reconstruction, ground_truth)  # [batch_size]
                        psnr_list = iter_psnr_values.cpu().tolist()
                        for img_idx_in_batch, psnr_val in enumerate(psnr_list):
                            global_img_idx = num_images_processed + img_idx_in_batch
                            psnr_per_image_iter[global_img_idx].append(psnr_val)
                                                
                        # Time for this iteration
                        iter_time = time.perf_counter() - iter_start
                        batch_time_list.append(iter_time)
                        
                        # Capture GPU metrics
                        iter_metrics = self.gpu_tracker.capture_iteration_result()
                        iter_metrics['iteration_time_sec'] = iter_time
                        
                        # Initialize dict structure on first use and store values
                        for key, val in iter_metrics.items():
                            if key not in all_batch_iter_metrics:
                                all_batch_iter_metrics[key] = []
                            if len(all_batch_iter_metrics[key]) == batch_idx:
                                all_batch_iter_metrics[key].append([])
                            all_batch_iter_metrics[key][batch_idx].append(val)
                        
                        print(f"Batch {batch_idx}, Iter {iter_idx + 1}/{self.n_iter}: PSNR = {iter_psnr_values.mean().item():.2f} dB, Time = {iter_time:.3f}s")
                    
                    # Store iteration times for this batch
                    time_per_batch_iter.append(batch_time_list)
                    
                    # Update count of images processed
                    num_images_processed += batch_size
                
                # Store results
                self.psnr_history.append((init_psnr_per_image, psnr_per_image_iter))
                self.iter_times.append((init_time_per_batch, time_per_batch_iter))  # (init_times[num_batches], iter_times[num_batches][n_iter])
                self.all_results.append(all_batch_iter_metrics)
            
            # Return final PSNR (average of last iteration across all images)
            _, psnr_iters = self.psnr_history[-1]
            final_psnr = sum(img[-1] for img in psnr_iters) / len(psnr_iters)
            return final_psnr
            
    def _run_with_context(self, cb, ctx=None):
        """Run PnP with optional distributed context.

        This unified method handles both single-process and distributed execution.

        Args:
            cb: Callback function
            ctx: Optional distributed context (None for single-process)
        """

        # Determine device
        if ctx is not None:
            self.device = ctx.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize GPU metrics tracker
        self.gpu_tracker = GPUMetricsTracker(device=self.device)

        # Track PSNR history and per-iteration metrics
        self.psnr_history = []
        self.iter_times = []
        self.psnr_metric = PSNR(reduction="none")  # Get per-image PSNR values
        

        # Initialize reconstructions list
        self.reconstructions = []

        # Handle physics: can be stacked physics, factory function, or list
        if ctx is not None and self.distribute_physics:
            # In distributed mode with distribute_physics=True
            physics = distribute(
                self.physics,
                ctx,
                num_operators=self.num_operators,
                type_object="linear_physics",
            )
        elif callable(self.physics) and not isinstance(self.physics, Physics):
            # Factory function in single-process mode: instantiate all operators and stack them
            physics_list = []
            for i in range(self.num_operators):
                op = self.physics(i, self.device, None)
                physics_list.append(op)
            physics = stack(*physics_list)
        else:
            # Already a stacked physics or single physics operator
            physics = self.physics
            if hasattr(physics, "to"):
                physics = physics.to(self.device)

         # Setup components once
        prior, data_fidelity = self._setup_components(self.device, ctx)
        print("Components set up.")

        # Compute step size
        step_size = self._compute_step_size(physics, self.device)
        print("Step size computed:", step_size)

        # Run PnP iterations (now loads measurements from dataloader each run)
        final_psnr = self._run_pnp_iterations(
            prior, data_fidelity, physics, step_size, cb, ctx
        )
        print(f"Final average PSNR: {final_psnr:.2f} dB")

        # Save results to CSV file 
        if self.all_results:               
                save_result_per_rank(self.all_results, self.name, self.max_batch_size)

        # Synchronize at the end of the run to ensure benchopt captures the full execution time
        # This must be done BEFORE the context manager exits

        if ctx is not None:
            ctx.barrier()


    def get_result(self):
        """Return the reconstruction result.

        Returns:
            dict: Dictionary with scalar PSNR and per-iteration data:
            - avg_psnr: Final PSNR (scalar) - for objective
            - init_psnr_per_image: [num_images] - initial PSNR values
            - psnr_per_image_iter: [num_images][n_iter] - iteration PSNR values
            - init_time_per_batch: [num_batches] - init time per batch (1D array)
            - iter_times_per_batch: [num_batches][n_iter] - iteration times (2D array)
            - gpu metrics: [num_batches][n_iter] - GPU metrics for iterations (2D arrays)
        """
        if not self.psnr_history:
            return dict(avg_psnr=0.0, name=self.name)
        
        # Unpack results
        init_psnr, psnr_iters = self.psnr_history[-1]
        init_time, time_iters = self.iter_times[-1]
        
        # Final PSNR is average of last iteration across all images
        final_psnr = sum(img[-1] for img in psnr_iters) / len(psnr_iters)
        
        result = dict(
            avg_psnr=final_psnr,
            init_psnr_per_image=init_psnr,
            psnr_per_image_iter=psnr_iters,
            init_time_per_batch=init_time,
            iter_times_per_batch=time_iters,
            name=self.name
        )
        
        # Add GPU metrics [batch][n_iter+1]
        if self.all_results:
            result.update(self.all_results[-1])

        return result

    def get_next(self, stop_val):
        return stop_val + 1
