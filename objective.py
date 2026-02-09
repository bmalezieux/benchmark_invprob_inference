"""Reconstruction objective for inverse problems benchmarking.

This objective evaluates reconstruction quality using PSNR and SSIM metrics,
and optionally saves comparison figures for visual inspection.
"""
import torch
from pathlib import Path

from benchopt import BaseObjective
from deepinv.loss.metric import PSNR, SSIM
from benchmark_utils import save_comparison_figure



class Objective(BaseObjective):
    """Reconstruction objective for inverse problems.
    
    Evaluates reconstruction quality using PSNR and SSIM metrics.
    Optionally saves comparison figures for visual inspection.
    """
    name = 'reconstruction_objective'

    # The three methods below define the links between the Dataset,
    # the Objective and the Solver.
    def set_data(self, ground_truth, measurement, physics, min_pixel=0.0, max_pixel=1.0, 
                 ground_truth_shape=None, num_operators=None):
        """Set the data from a Dataset to compute the objective.

        Parameters
        ----------
        ground_truth : torch.Tensor
            Ground truth image.
        measurement : torch.Tensor or TensorList
            Noisy measurements.
        physics : Physics
            Forward operator.
        min_pixel : float, optional
            Minimum pixel value for metrics. Default: 0.0.
        max_pixel : float, optional
            Maximum pixel value for metrics. Default: 1.0.
        ground_truth_shape : tuple, optional
            Shape of ground truth tensor.
        num_operators : int, optional
            Number of operators in stacked physics.
        """
        self.ground_truth = ground_truth
        self.measurement = measurement
        self.physics = physics
        self.ground_truth_shape = ground_truth_shape if ground_truth_shape is not None else ground_truth.shape
        self.num_operators = num_operators if num_operators is not None else 1
        self.psnr_metric = PSNR(max_pixel=max_pixel)
        # self.ssim_metric = SSIM(max_pixel=max_pixel)
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.evaluation_count = 0

    def get_objective(self):
        """Returns a dict passed to Solver.set_objective method.
        
        Returns
        -------
        dict
            Dictionary with measurement, physics, and metadata.
        """
        return dict(
            measurement=self.measurement, 
            physics=self.physics,
            ground_truth_shape=self.ground_truth_shape,
            num_operators=self.num_operators,
            min_pixel=self.min_pixel,
            max_pixel=self.max_pixel,
        )

    def evaluate_result(self, reconstruction, name, **kwargs):
        """Compute the objective value(s) given the output of a solver.

        Parameters
        ----------
        reconstruction : torch.Tensor
            Reconstructed image from solver.
        name : str
            Name identifier for the solver/configuration.
        **kwargs : dict
            Optional GPU and step metrics including:
            - gpu_memory_allocated_mb, gpu_memory_reserved_mb, 
              gpu_memory_max_allocated_mb, gpu_available_memory_mb
            - gradient_time_sec, gradient_memory_allocated_mb,
              gradient_memory_reserved_mb, gradient_memory_delta_mb,
              gradient_memory_peak_mb
            - denoise_time_sec, denoise_memory_allocated_mb,
              denoise_memory_reserved_mb, denoise_memory_delta_mb,
              denoise_memory_peak_mb
            
        Returns
        -------
        dict
            Dictionary with 'value' (negative PSNR for minimization),
            'psnr', and optional GPU/step metrics.
        """
        with torch.no_grad():
            # Ensure reconstruction is on the same device as ground truth
            reconstruction = reconstruction.to(self.ground_truth.device)

            psnr_tensor = self.psnr_metric(reconstruction, self.ground_truth)
            # ssim_tensor = self.ssim_metric(reconstruction, self.ground_truth)

            # Handle batch case - take mean across batch dimension
            psnr = (
                psnr_tensor.mean().item()
                if psnr_tensor.numel() > 1
                else psnr_tensor.item()
            )
            # ssim = (
            #     ssim_tensor.mean().item()
            #     if ssim_tensor.numel() > 1
            #     else ssim_tensor.item()
            # )

            # Save comparison figure
            output_dir = "evaluation_output/" + name.replace('/', '_').replace('..', '')
            self.evaluation_count += 1
            save_comparison_figure(
                self.ground_truth, 
                reconstruction,
                # metrics={'psnr': psnr, 'ssim': ssim},
                metrics={'psnr': psnr},
                output_dir=output_dir,
                filename=f'eval_{self.evaluation_count:04d}.png',
                evaluation_count=self.evaluation_count
            )

        # Return value (primary metric for stopping criterion) and additional metrics
        result = dict(value=-psnr, psnr=psnr)
        
        # Add all non-None metrics from kwargs to result
        for key, value in kwargs.items():
            if value is not None:
                result[key] = value
        
        return result

    def get_one_result(self):
        """Return one solution for which the objective can be evaluated.

        This function is mostly used for testing and debugging purposes.
        
        Returns
        -------
        dict
            Dictionary with a noisy version of ground truth.
        """
        return dict(
            reconstruction=self.ground_truth + self.ground_truth.std(),
            name="test_result"
        )