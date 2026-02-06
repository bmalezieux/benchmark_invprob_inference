import os
from datetime import datetime
import torch
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.distributed import DistributedContext, distribute
from deepinv.physics import Physics, stack
from deepinv.utils.tensorlist import TensorList

from benchopt import BaseSolver
from benchmark_utils import create_drunet_denoiser, GPUMetricsTracker, save_result_per_rank


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
        x_example = torch.zeros_like(ground_truth, device=ground_truth.device, dtype=ground_truth.dtype)

        # Compute Lipschitz constant
        lipschitz_constant = operator.compute_norm(x_example)
        
        return 1.0 / lipschitz_constant if lipschitz_constant > 0 else 1.0


def initialize_reconstruction(
    signal_shape: tuple,
    operator: Physics,
    measurements,
    device: torch.device,
    method: str = 'pseudo_inverse',
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
    if method == 'zeros':
        return torch.zeros(signal_shape, device=device)

    elif method == 'pseudo_inverse':
        x_init = operator.A_dagger(measurements).clamp(0, None)
        return x_init

    else:
        raise ValueError(f"Unknown initialization method: {method}. Use 'zeros' or 'pseudo_inverse'")



class Solver(BaseSolver):
    """Plug-and-Play (PnP) solver with optional distributed support."""
    
    name = 'PnP'
    
    # Use callback sampling strategy for transparent iteration control
    sampling_strategy = 'callback'
    
    # Solver parameters
    parameters = {
        'denoiser': ['drunet'],
        'denoiser_lambda_relaxation': [None],
        'step_size': [None],
        'step_size_scale': [0.99],
        'denoiser_sigma': [0.05],
        'distribute_physics': [False],
        'distribute_denoiser': [False],
        'patch_size': [128],
        'overlap': [32],
        'max_batch_size': [0],
        'init_method': ['pseudo_inverse'],
        'slurm_nodes': [1],
        'slurm_ntasks_per_node': [1],
        'slurm_gres': ["gpu:1"],
        'torchrun_nproc_per_node': [1],
        'name_prefix': ['pnp'],
    }

    def set_objective(self, measurement, physics, ground_truth_shape, num_operators, min_pixel=0.0, max_pixel=1.0):
        """Set the objective from the dataset.
        
        Args:
            measurement: Noisy measurements (TensorList or tensor)
            physics: Forward operator (stacked physics or list or callable)
            ground_truth_shape: Shape of the ground truth tensor
            num_operators: Number of operators in the physics
        """
        self.measurement = measurement
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
            print(f"Distributed environment already initialized: world_size={self.world_size}")
        else:
            # Try to initialize
            try:
                import submitit
                submitit.helpers.TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
                self.world_size = int(os.environ.get("WORLD_SIZE", 1))
                print(f"Initialized distributed environment via submitit: world_size={self.world_size}")
            except ImportError:
                print("submitit not installed, running in non-distributed mode")
            except RuntimeError as e:
                # This could be SLURM not available or other runtime issues
                error_msg = str(e).lower()
                if "slurm" in error_msg or "environment" in error_msg:
                    print(f"SLURM environment not available: {e}")
                else:
                    print(f"RuntimeError initializing submitit (possibly already called): {e}")
                print("Running in non-distributed mode")

        self.distributed_mode = self.world_size > 1
        self.reconstruction = torch.zeros(self.ground_truth_shape)
        
        # Initialize GPU metrics tracker (device will be set in _run_with_context)
        self.gpu_tracker = None
        
        # Initialize results storage for per-iteration tracking
        self.all_results = []
        
        # Generate name based on whether using slurm or torchrun
        if hasattr(self, 'slurm_ntasks_per_node') and self.slurm_ntasks_per_node > 1:
            self.name = self.name_prefix + datetime.now().strftime("_%Y%m%d_%H%M%S_") + f"{self.slurm_nodes}n{self.slurm_ntasks_per_node}t"
        elif hasattr(self, 'torchrun_nproc_per_node') and self.torchrun_nproc_per_node > 1:
            self.name = self.name_prefix + datetime.now().strftime("_%Y%m%d_%H%M%S_") + f"torchrun_{self.torchrun_nproc_per_node}proc"
        else:
            self.name = self.name_prefix + datetime.now().strftime("_%Y%m%d_%H%M%S_") + "_single"
        
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
        if self.denoiser == 'drunet':
            # Use the utility function to create appropriate DRUNet
            denoiser = create_drunet_denoiser(
                ground_truth_shape=self.ground_truth_shape,
                device=device,
                dtype=torch.float32
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
                tiling_dims=(-3,-2,-1) if len(self.ground_truth_shape) == 5 else (-2,-1),
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
    
    def _initialize_reconstruction(self, physics, measurements, device):
        """Initialize reconstruction signal.
        
        Args:
            physics: Physics operator (can be stacked or distributed)
            measurements: Measurements
            device: Device to use
            ctx: Optional distributed context
            
        Returns:
            Initialized reconstruction tensor
        """
        with torch.no_grad():
            return initialize_reconstruction(
                signal_shape=self.ground_truth_shape,
                operator=physics,
                measurements=measurements,
                device=device,
                method=self.init_method,
            )
    
    def _run_pnp_iterations(self, prior, data_fidelity, physics, measurements, step_size, cb):
        """Run PnP iterations.
        
        Args:
            x: Initial reconstruction
            prior: PnP prior
            data_fidelity: L2 data fidelity
            physics: Physics operators
            measurements: Measurements
            step_size: Step size for gradient descent
            cb: Callback function
            ctx: Optional distributed context
            
        Returns:
            Final reconstruction
        """
        with torch.no_grad():
            
            while True:
                keep_going = cb()

                if self.distributed_mode and self.ctx is not None:
                    # Synchronize stopping criterion
                    decision = torch.tensor([float(keep_going)], device=self.device)
                    self.ctx.broadcast(decision, src=0)
                    keep_going = bool(decision.item())

                if not keep_going:
                    break
                
                # Reset iteration tracking for new iteration
                self.gpu_tracker.reset_iteration_tracking()

                # ===== GRADIENT STEP =====
                with self.gpu_tracker.track_step('gradient'):
                    # Data fidelity gradient step
                    grad = data_fidelity.grad(self.reconstruction, measurements, physics)
                    # Gradient descent step
                    self.reconstruction = self.reconstruction - step_size * grad

                # ===== DENOISING STEP =====
                with self.gpu_tracker.track_step('denoise'):
                    # Denoising step
                    if self.denoiser_lambda_relaxation is None:
                        self.reconstruction = prior.prox(self.reconstruction, sigma_denoiser=self.denoiser_sigma)
                    else:
                        denoised_reconstruction = prior.prox(self.reconstruction, sigma_denoiser=self.denoiser_sigma)
                        lamda = self.denoiser_lambda_relaxation
                        alpha = (step_size * lamda) / (1 + step_size * lamda)
                        self.reconstruction = (1 - alpha) * self.reconstruction + alpha * denoised_reconstruction

                    # Clip reconstruction to valid range after denoising
                    if self.clip_range is not None:
                        self.reconstruction = torch.clamp(self.reconstruction, self.clip_range[0], self.clip_range[1])

                # Synchronize all CUDA operations and distributed processes
                # This ensures accurate timing measurements
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                
                # Capture iteration metrics after both steps complete
                iteration_result = self.gpu_tracker.capture_iteration_result()
                self.all_results.append(iteration_result)

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

        # Move measurement to correct device
        if hasattr(self.measurement, 'to'):
            measurement = self.measurement.to(self.device)
        elif isinstance(self.measurement, list):
            measurement = TensorList([m.to(self.device) for m in self.measurement])
        else:
            measurement = self.measurement

        # Handle physics: can be stacked physics, factory function, or list
        if ctx is not None and self.distribute_physics:
            # In distributed mode with distribute_physics=True
            physics = distribute(
                self.physics, 
                ctx, 
                num_operators=self.num_operators,
                type_object='linear_physics'
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
            if hasattr(physics, 'to'):
                physics = physics.to(self.device)
        
        # Setup components
        prior, data_fidelity = self._setup_components(self.device, ctx)
        #cb()
        print("Components set up.")
        
        # Compute step size
        step_size = self._compute_step_size(physics, self.device)
        #cb()
        print("Step size computed:", step_size)

        # Initialize reconstruction
        self.reconstruction = self._initialize_reconstruction(physics, self.measurement, self.device)
        #cb()
        print("Reconstruction initialized.")
        
        # Clip initial reconstruction if requested
        if self.clip_range is not None:
            self.reconstruction = torch.clamp(self.reconstruction, self.clip_range[0], self.clip_range[1])

        # Synchronize before capturing initial state
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            
        print("Starting PnP iterations.")

        # Run PnP iterations
        self._run_pnp_iterations(prior, data_fidelity, physics, measurement, step_size, cb)

        # Synchronize at the end of the run to ensure benchopt captures the full execution time
        # This must be done BEFORE the context manager exits
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        if ctx is not None:
            ctx.barrier()
        
        # Save results to file (both distributed and single-GPU modes)
        save_result_per_rank(self.all_results, self.name, self.max_batch_size)

    def get_result(self):
        """Return the reconstruction result.
        
        Returns:
            dict: Dictionary with 'reconstruction' key, GPU memory snapshot,
            and per-step metrics (gradient and denoise timing/memory)
        """
        result = dict(reconstruction=self.reconstruction, name=self.name)
        
        # Add GPU memory snapshot
        if self.gpu_tracker is not None:
            gpu_mem = self.gpu_tracker.get_gpu_memory_snapshot()
            result.update({
                'gpu_memory_allocated_mb': gpu_mem['allocated_mb'],
                'gpu_memory_reserved_mb': gpu_mem['reserved_mb'],
                'gpu_memory_max_allocated_mb': gpu_mem['max_allocated_mb'],
                'gpu_available_memory_mb': gpu_mem['available_mb'],
            })
            
            # Add per-step metrics
            all_step_metrics = self.gpu_tracker.get_all_step_metrics()
            for step_name, metrics in all_step_metrics.items():
                result[f'{step_name}_time_sec'] = metrics['time_sec']
                result[f'{step_name}_memory_allocated_mb'] = metrics['memory_allocated_mb']
                result[f'{step_name}_memory_reserved_mb'] = metrics['memory_reserved_mb']
                result[f'{step_name}_memory_delta_mb'] = metrics['memory_delta_mb']
                result[f'{step_name}_memory_peak_mb'] = metrics['memory_peak_mb']
        
        return result

    def get_next(self, stop_val): return stop_val + 1