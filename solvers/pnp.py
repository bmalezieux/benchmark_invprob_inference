import os
from datetime import datetime
import torch
from typing import List, Optional
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.models import DRUNet
from deepinv.distrib import DistributedContext, distribute
from deepinv.physics import Physics
from deepinv.utils.tensorlist import TensorList

from benchopt import BaseSolver


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
        x_init = operator.A_dagger(measurements)
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
        'step_size': [None],
        'step_size_scale': [0.9],
        'denoiser_sigma': [0.05],
        'distribute_physics': [False],
        'distribute_denoiser': [False],
        'patch_size': [128],
        'receptive_field_size': [32],
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
            physics: Forward operator (stacked physics or list)
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

        try:
            import submitit
            submitit.helpers.TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        except (ImportError, RuntimeError):
            pass

        self.distributed_mode = self.world_size > 1
        self.reconstruction = torch.zeros(self.ground_truth_shape)
        
        # Generate name based on whether using slurm or torchrun
        if hasattr(self, 'slurm_tasks_per_node') and self.slurm_tasks_per_node > 1:
            self.name = self.name_prefix + datetime.now().strftime("_%Y%m%d_%H%M%S_") + f"{self.slurm_nodes}n{self.slurm_tasks_per_node}t"
        elif hasattr(self, 'torchrun_nproc_per_node') and self.torchrun_nproc_per_node > 1:
            self.name = self.name_prefix + datetime.now().strftime("_%Y%m%d_%H%M%S_") + f"torchrun_{self.torchrun_nproc_per_node}proc"
        else:
            self.name = self.name_prefix + datetime.now().strftime("_%Y%m%d_%H%M%S_") + "_single"
    
    def run(self, cb):
        """Run the PnP algorithm with callback for iteration control.
        
        Args:
            cb: Callback function to call at each iteration. Returns False when to stop.
        """
        if self.distributed_mode:
            with DistributedContext(seed=42) as ctx:
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
            denoiser = DRUNet(pretrained="download").to(device)
        else:
            raise ValueError(f"Unknown denoiser: {self.denoiser}")
                
        # Distribute denoiser if context provided and requested
        if ctx is not None and self.distribute_denoiser:
            denoiser = distribute(
                denoiser,
                ctx,
                patch_size=self.patch_size,
                receptive_field_size=self.receptive_field_size,
                tiling_dims=2,
                max_batch_size=self.max_batch_size,
            )
        
        # Create prior and data fidelity
        prior = PnP(denoiser=denoiser)
        data_fidelity = L2()
        
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
            
            while cb():
                # Data fidelity gradient step
                grad = data_fidelity.grad(self.reconstruction, measurements, physics)

                # Gradient descent step
                self.reconstruction = self.reconstruction - step_size * grad

                # Denoising step
                self.reconstruction = prior.prox(self.reconstruction, sigma_denoiser=self.denoiser_sigma)

                # Clip reconstruction to valid range after denoising
                if self.clip_range is not None:
                    self.reconstruction = torch.clamp(self.reconstruction, self.clip_range[0], self.clip_range[1])

                # Synchronize all CUDA operations and distributed processes
                # This ensures accurate timing measurements
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)

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

        # Move measurement to correct device
        if hasattr(self.measurement, 'to'):
            measurement = self.measurement.to(self.device)
        elif isinstance(self.measurement, list):
            measurement = TensorList([m.to(self.device) for m in self.measurement])
        else:
            measurement = self.measurement

        # Distribute physics if in distributed mode and requested
        if ctx is not None and self.distribute_physics:
            physics = distribute(self.physics, ctx)
        else:
            physics = self.physics
            if hasattr(physics, 'to'):
                physics = physics.to(self.device)
        
        # Setup components
        prior, data_fidelity = self._setup_components(self.device, ctx)
        cb()
        print("Components set up.")
        
        # Compute step size
        step_size = self._compute_step_size(physics, self.device)
        cb()
        print("Step size computed:", step_size)

        # Initialize reconstruction
        self.reconstruction = self._initialize_reconstruction(physics, self.measurement, self.device)
        cb()
        print("Reconstruction initialized.")
        
        # Clip initial reconstruction if requested
        if self.clip_range is not None:
            self.reconstruction = torch.clamp(self.reconstruction, self.clip_range[0], self.clip_range[1])

        print("Starting PnP iterations.")

        # Run PnP iterations
        self._run_pnp_iterations(prior, data_fidelity, physics, measurement, step_size, cb)

        # Synchronize at the end of the run to ensure benchopt captures the full execution time
        # This must be done BEFORE the context manager exits
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        if ctx is not None:
            ctx.barrier()

    def get_result(self):
        """Return the reconstruction result.
        
        Returns:
            dict: Dictionary with 'reconstruction' key
        """
        return dict(reconstruction=self.reconstruction, name=self.name)
