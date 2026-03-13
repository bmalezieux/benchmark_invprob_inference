import os
from datetime import datetime

import torch
from benchopt import BaseSolver
from deepinv.distributed import DistributedContext, distribute
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optim_iterators import PGDIteration
from deepinv.optim.prior import PnP
from deepinv.physics import Physics, stack
from deepinv.utils.tensorlist import TensorList

from benchmark_utils import create_drunet_denoiser
from benchmark_utils.gpu_metrics import GPUMetricsTracker, save_result_per_rank


def compute_step_size_from_operator(
    operator: Physics,
    ground_truth: torch.Tensor,
) -> float:
    """
    Compute step size from Lipschitz constant of operator.

    Args:
        operator: Physics operator (can be stacked or distributed)
        ground_truth: Ground truth tensor (used for creating example signal)

    Returns:
        Step size (1 / lipschitz_constant)
    """
    with torch.no_grad():
        x_example = torch.zeros_like(
            ground_truth, device=ground_truth.device, dtype=ground_truth.dtype
        )
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
        x_init = operator.A_dagger(measurements)
        x_init = x_init.clamp(0, 1)
        return x_init

    else:
        raise ValueError(
            f"Unknown initialization method: {method}. Use 'zeros' or 'pseudo_inverse'"
        )


class Solver(BaseSolver):
    """Plug-and-Play PGD solver: uses deepinv's PGDIteration with a DRUNet denoiser as prior.

    Implements the iteration via PGDIteration:

        u_k   = x_k - step_size * grad_f(x_k)
        x_{k+1} = D_sigma(u_k)   (denoiser as proximal operator)

    where f is the L2 data-fidelity and the prior is PnP with a DRUNet denoiser.
    """

    name = "PGD"

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
        "denoiser_sigma": [0.05],
        "step_size": [None],
        "step_size_scale": [0.99],
        "distribute_physics": [False],
        "distribute_denoiser": [False],
        "torch_compile": [False],
        "patch_size": [128],
        "overlap": [32],
        "max_batch_size": [0],
        "init_method": ["pseudo_inverse"],
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
        "torchrun_nproc_per_node": [1],
        "name_prefix": ["pgd"],
    }

    def set_objective(
        self,
        measurement,
        physics,
        ground_truth_shape,
        num_operators,
        min_pixel=0.0,
        max_pixel=1.0,
    ):
        """Set the objective from the dataset.

        Args:
            measurement: Noisy measurements (TensorList or tensor)
            physics: Forward operator (stacked physics or factory function)
            ground_truth_shape: Shape of the ground truth tensor
            num_operators: Number of operators in the physics
            min_pixel: Minimum pixel value for clipping
            max_pixel: Maximum pixel value for clipping
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
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            print(
                f"Distributed environment already initialized: world_size={self.world_size}"
            )
        else:
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
                error_msg = str(e).lower()
                if "slurm" in error_msg or "environment" in error_msg:
                    print(f"SLURM environment not available: {e}")
                else:
                    print(
                        f"RuntimeError initializing submitit (possibly already called): {e}"
                    )
                print("Running in non-distributed mode")

        self.distributed_mode = self.world_size > 1
        self.reconstruction = torch.zeros(self.ground_truth_shape)

        self.gpu_tracker = None
        self.all_results = []

        # Generate run name
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

        if self.distributed_mode:
            self.name = self.name + f"_rank{int(os.environ.get('RANK', 0))}"

    def run(self, cb):
        """Run the PGD algorithm with callback for iteration control.

        Args:
            cb: Callback function to call at each iteration. Returns False when to stop.
        """
        if self.distributed_mode:
            with DistributedContext(seed=42, cleanup=True) as ctx:
                self.ctx = ctx
                self._run_with_context(cb, ctx)
        else:
            self._run_with_context(cb, ctx=None)

    def _setup_components(self, device, ctx=None):
        """Build the denoiser, prior, data-fidelity, and PGD iterator.

        Args:
            device: Torch device to use.
            ctx: Optional distributed context.

        Returns:
            Tuple of (pgd_iter, prior, data_fidelity)
        """
        if self.denoiser == "drunet":
            denoiser = create_drunet_denoiser(
                ground_truth_shape=self.ground_truth_shape,
                device=device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unknown denoiser: {self.denoiser}")

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

        # Optionally compile denoiser before wrapping in prior
        if self.torch_compile:
            denoiser = torch.compile(denoiser)

        prior = PnP(denoiser=denoiser)
        data_fidelity = L2()

        if ctx is not None and self.distribute_physics:
            data_fidelity = distribute(data_fidelity, ctx)

        if self.torch_compile:
            data_fidelity = torch.compile(data_fidelity)

        pgd_iter = PGDIteration(has_cost=False)

        return pgd_iter, prior, data_fidelity

    def _compute_step_size(self, physics, device):
        """Compute step size from physics operator Lipschitz constant.

        Args:
            physics: Physics operator
            device: Torch device

        Returns:
            Step size value
        """
        if isinstance(self.step_size, float):
            return self.step_size

        ground_truth_example = torch.zeros(self.ground_truth_shape, device=device)
        raw_step_size = compute_step_size_from_operator(physics, ground_truth_example)
        return raw_step_size * self.step_size_scale

    def _initialize_reconstruction(self, physics, measurements, device):
        """Initialize the reconstruction.

        Args:
            physics: Physics operator
            measurements: Measurements
            device: Torch device

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

    def _run_pgd_iterations(
        self, pgd_iter, prior, data_fidelity, physics, measurements, step_size, cb
    ):
        """Run PGD iterations using deepinv's PGDIteration.


        Args:
            pgd_iter: PGDIteration instance from deepinv
            prior: Pnp Prior instance 
            data_fidelity: L2 data fidelity
            physics: Physics operators
            measurements: Measurements
            step_size: Gradient descent step size
            cb: Benchopt callback
        """
        if self.denoiser_lambda_relaxation is not None:
            lamda = self.denoiser_lambda_relaxation
            beta = (step_size * lamda) / (1 + step_size * lamda)
        else:
            beta = 1.0

        cur_params = {
            "stepsize": step_size,
            "lambda": 1.0,
            "g_param": self.denoiser_sigma,
            "beta": beta,
        }

        with torch.no_grad():

            while True:
                keep_going = cb()

                if self.distributed_mode and self.ctx is not None:
                    # Synchronize stopping criterion across ranks
                    decision = torch.tensor([float(keep_going)], device=self.device)
                    self.ctx.broadcast(decision, src=0)
                    keep_going = bool(decision.item())

                if not keep_going:
                    break

                self.gpu_tracker.reset_iteration_tracking()

                # ===== PGD STEP (gradient + proximal via deepinv) =====
                with self.gpu_tracker.track_step("pgd"):
                    X = {"est": [self.reconstruction]}
                    X = pgd_iter(X, data_fidelity, prior, cur_params, measurements, physics)
                    self.reconstruction = X["est"][0]

                    # Clip to valid pixel range
                    if self.clip_range is not None:
                        self.reconstruction = torch.clamp(
                            self.reconstruction, self.clip_range[0], self.clip_range[1]
                        )

                # Synchronize CUDA for accurate timing
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

                iteration_result = self.gpu_tracker.capture_iteration_result()
                self.all_results.append(iteration_result)

    def _run_with_context(self, cb, ctx=None):
        """Run PGD with optional distributed context.

        Args:
            cb: Benchopt callback
            ctx: Optional DistributedContext (None for single-process)
        """
        # Determine device
        if ctx is not None:
            self.device = ctx.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize GPU metrics tracker
        self.gpu_tracker = GPUMetricsTracker(device=self.device)

        # Move measurements to device
        if hasattr(self.measurement, "to"):
            measurement = self.measurement.to(self.device)
        elif isinstance(self.measurement, list):
            measurement = TensorList([m.to(self.device) for m in self.measurement])
        else:
            measurement = self.measurement

        # Prepare physics
        if ctx is not None and self.distribute_physics:
            physics = distribute(
                self.physics,
                ctx,
                num_operators=self.num_operators,
                type_object="linear_physics",
            )
        elif callable(self.physics) and not isinstance(self.physics, Physics):
            # Factory function → instantiate and stack all operators
            physics_list = [
                self.physics(i, self.device, None) for i in range(self.num_operators)
            ]
            physics = stack(*physics_list)
        else:
            physics = self.physics
            if hasattr(physics, "to"):
                physics = physics.to(self.device)

        # Build components
        pgd_iter, prior, data_fidelity = self._setup_components(self.device, ctx)
        print("Components set up.")

        # Compute step size
        step_size = self._compute_step_size(physics, self.device)
        print("Step size computed:", step_size)

        # Initialize reconstruction
        self.reconstruction = self._initialize_reconstruction(
            physics, measurement, self.device
        )
        print("Reconstruction initialized.")

        if self.clip_range is not None:
            self.reconstruction = torch.clamp(
                self.reconstruction, self.clip_range[0], self.clip_range[1]
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        print("Starting PGD iterations.")

        # Run iterations
        self._run_pgd_iterations(
            pgd_iter, prior, data_fidelity, physics, measurement, step_size, cb
        )

        # Final synchronization before context exit
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if ctx is not None:
            ctx.barrier()

        # Save per-rank results to file
        save_result_per_rank(self.all_results, self.name, self.max_batch_size)

    def get_result(self):
        """Return the reconstruction result with GPU and timing metrics.

        Returns:
            dict with 'reconstruction', 'name', GPU memory stats,
            and per-step timing/memory metrics (pgd).
        """
        result = dict(reconstruction=self.reconstruction, name=self.name)

        if self.gpu_tracker is not None:
            gpu_mem = self.gpu_tracker.get_gpu_memory_snapshot()
            result.update(
                {
                    "gpu_memory_allocated_mb": gpu_mem["allocated_mb"],
                    "gpu_memory_reserved_mb": gpu_mem["reserved_mb"],
                    "gpu_memory_max_allocated_mb": gpu_mem["max_allocated_mb"],
                    "gpu_available_memory_mb": gpu_mem["available_mb"],
                }
            )

            all_step_metrics = self.gpu_tracker.get_all_step_metrics()
            for step_name, metrics in all_step_metrics.items():
                result[f"{step_name}_time_sec"] = metrics["time_sec"]
                result[f"{step_name}_memory_allocated_mb"] = metrics[
                    "memory_allocated_mb"
                ]
                result[f"{step_name}_memory_reserved_mb"] = metrics[
                    "memory_reserved_mb"
                ]
                result[f"{step_name}_memory_delta_mb"] = metrics["memory_delta_mb"]
                result[f"{step_name}_memory_peak_mb"] = metrics["memory_peak_mb"]

        return result

    def get_next(self, stop_val):
        return stop_val + 1
