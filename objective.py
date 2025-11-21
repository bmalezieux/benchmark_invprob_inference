import torch
import matplotlib.pyplot as plt
from pathlib import Path

from benchopt import BaseObjective
from deepinv.loss.metric import PSNR, SSIM



class Objective(BaseObjective):
    # Name of the Objective function
    name = 'reconstruction_objective'

    # The three methods below define the links between the Dataset,
    # the Objective and the Solver.
    def set_data(self, ground_truth, measurement, physics, min_pixel=0.0, max_pixel=1.0, 
                 ground_truth_shape=None, num_operators=None):
        """Set the data from a Dataset to compute the objective.

        The argument are the keys of the dictionary returned by
        ``Dataset.get_data``.
        """
        self.ground_truth = ground_truth
        self.measurement = measurement
        self.physics = physics
        self.ground_truth_shape = ground_truth_shape if ground_truth_shape is not None else ground_truth.shape
        self.num_operators = num_operators if num_operators is not None else 1
        self.psnr_metric = PSNR(max_pixel=max_pixel)
        self.ssim_metric = SSIM(max_pixel=max_pixel)
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.evaluation_count = 0

    def get_objective(self):
        "Returns a dict passed to ``Solver.set_objective`` method."
        return dict(
            measurement=self.measurement, 
            physics=self.physics,
            ground_truth_shape=self.ground_truth_shape,
            num_operators=self.num_operators,
            min_pixel=self.min_pixel,
            max_pixel=self.max_pixel,
        )

    def evaluate_result(self, reconstruction, name):
        """Compute the objective value(s) given the output of a solver.

        The arguments are the keys in the dictionary returned
        by ``Solver.get_result``.
        """

        with torch.no_grad():
            # Ensure reconstruction is on the same device as ground truth
            reconstruction = reconstruction.to(self.ground_truth.device)
            
            psnr_tensor = self.psnr_metric(reconstruction, self.ground_truth)
            ssim_tensor = self.ssim_metric(reconstruction, self.ground_truth)

            # Handle batch case - take mean across batch dimension
            psnr = (
                psnr_tensor.mean().item()
                if psnr_tensor.numel() > 1
                else psnr_tensor.item()
            )
            ssim = (
                ssim_tensor.mean().item()
                if ssim_tensor.numel() > 1
                else ssim_tensor.item()
            )
            
            # Save comparison figure
            self._save_comparison_figure(reconstruction, psnr, ssim, output_dir="evaluation_output/" + name)
            self.evaluation_count += 1

        # Return value (primary metric for stopping criterion) and additional metrics
        # Use PSNR as the primary metric (higher is better)
        return dict(value=-psnr, psnr=psnr, ssim=ssim)
    
    def _save_comparison_figure(self, reconstruction, psnr, ssim, output_dir="evaluation_output"):
        """Save a comparison figure showing ground truth and reconstruction side by side.
        
        Parameters
        ----------
        reconstruction : torch.Tensor
            Reconstruction tensor.
        psnr : float
            PSNR value.
        ssim : float
            SSIM value.
        output_dir : str
            Directory to save the comparison figure.
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to numpy for visualization
        def tensor_to_numpy(tensor):
            """Convert tensor to numpy array suitable for imshow."""
            img = tensor.detach().cpu()
            if img.ndim == 4:
                img = img.squeeze(0)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            img = torch.clamp(img, 0, 1)
            return img.numpy()
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot ground truth
        gt_img = tensor_to_numpy(self.ground_truth)
        axes[0].imshow(gt_img, cmap='gray' if gt_img.ndim == 2 else None)
        axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Plot reconstruction
        recon_img = tensor_to_numpy(reconstruction)
        axes[1].imshow(recon_img, cmap='gray' if recon_img.ndim == 2 else None)
        axes[1].set_title(f'Reconstruction\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Add overall title with evaluation count
        fig.suptitle(f'Evaluation #{self.evaluation_count}', fontsize=16, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        output_file = output_path / f'eval_{self.evaluation_count:04d}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def get_one_result(self):
        """Return one solution for which the objective can be evaluated.

        This function is mostly used for testing and debugging purposes.
        """
        return dict(reconstruction=self.ground_truth + self.ground_truth.std())