import math
import numpy as np
import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
from astropy import constants as const
from deepinv.physics import RadioInterferometry
from typing_extensions import TypeAlias
from pathlib import Path

FilePathType: TypeAlias = Union[Path, str]


@dataclass
class DirtyImagerConfig:
    """Base class for the config / parameters of a dirty imager.

    Contains basic parameters common across all dirty imagers.
    Inherit and add parameters specific to a dirty imager implementation.

    Attributes:
        imaging_npixel (int): Image size
        imaging_cellsize (float): Scale of a pixel in radians
        combine_across_frequencies (bool): Whether or not to combine images
            across all frequency channels into one image. Defaults to True.

    """

    imaging_npixel: int
    imaging_cellsize: float
    binning_factor: float = 1.25
    nufft_k_oversampling: float = 1.5
    combine_across_frequencies: bool = True


class DeepinvDirtyImager(torch.nn.Module):
    """Dirty imager based on the DeepInv library.

    Attributes:
        config (DirtyImagerConfig): Config containing parameters for
            dirty imaging

    """

    def __init__(
        self, config: DirtyImagerConfig, device=torch.device("cpu"), verbose: int = 0
    ) -> None:
        """Initializes the instance with a config.

        Args:
            config (DirtyImagerConfig): see config attribute
            device (torch.device): Device to use for computations. Default is torch.device("cpu").

        """
        super().__init__()
        self.config = config
        self.device = device  # setup_device()  # Ensure device is set up correctly
        self.verbose = verbose

    def to_device(self, tensor, non_blocking=True, pin_memory=False):
        """Transfer tensor to device with optimized settings and error handling."""
        try:
            if self.device.type == "cuda":
                # Pin memory for faster transfers if specified
                if pin_memory and tensor.device.type == "cpu":
                    tensor = tensor.pin_memory()
                return tensor.to(self.device, non_blocking=non_blocking)
            else:
                return tensor.to(self.device)
        except RuntimeError as e:
            print(f"Transfer error to {self.device}: {e}")
            print("Falling back to CPU")
            return tensor.to("cpu")

    def load_visibilities(
        self,
        visibility_path: str,
        visibility_format: str = "MS",
        visibility_column: str = "DATA",
        /,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load data from MS file and convert to PyTorch tensors

        Args:
            visibility_path (str): Path to the visibility data file
            visibility_format (str): Format of the visibility data. Currently only "MS" is supported.
            visibility_column (str): Column name in the MS file containing the visibility data. Default is for "DATA". for the OSKAR simulator.

        """
        from casacore.tables import table

        if visibility_format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility_format} not supported, "
                "only MS format is currently supported"
            )
        # Get UVW coords and visibilities
        with table(visibility_path, readonly=True) as tb:

            # Direct loading with proper type
            uvw_np = tb.getcol("UVW").astype(np.float32)
            visibilities_np = tb.getcol(visibility_column).astype(np.complex64)

            print(f"Data loaded: {uvw_np.shape[0]} visibilities")
            print(f"Available columns: {tb.colnames()}")

        # Load frequencies
        with table(visibility_path + "/SPECTRAL_WINDOW", readonly=True) as tb:
            chan_freqs_np = tb.getcol("CHAN_FREQ")[0].astype(np.float32)

        print(f"Number of channels: {len(chan_freqs_np)}")

        # Optimized conversion to PyTorch with direct transfer to device
        uvw = self.to_device(torch.from_numpy(uvw_np))
        visibilities = self.to_device(torch.from_numpy(visibilities_np))
        freqs = self.to_device(torch.from_numpy(chan_freqs_np))

        # CPU memory cleanup
        del uvw_np, visibilities_np, chan_freqs_np

        return uvw, visibilities, freqs

    def normalize_uv_coords(
        self,
        uvw: torch.Tensor,
        freqs: torch.Tensor,
        visibilities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize UV coordinates and concatenate visibilities for NUFFT

        Optimizations:
        - Vectorized calculations
        - Tensor pre-allocation
        - Use of pre-calculated constants
        """

        # Stokes I polarization (vectorized)
        if visibilities.shape[2] == 4:
            visibilities = 0.5 * (visibilities[:, :, 0] + visibilities[:, :, 3])
        elif visibilities.shape[2] == 1:
            visibilities = visibilities[:, :, 0]

        n_vis, n_freq = visibilities.shape
        im_size = torch.tensor(
            [self.config.imaging_npixel, self.config.imaging_npixel], device=self.device
        )

        # Pre-allocation for better performance
        total_points = n_vis * n_freq
        samples_locs = torch.zeros(
            (2, total_points), dtype=torch.float32, device=self.device
        )
        all_visibilities = torch.zeros(
            total_points, dtype=torch.complex64, device=self.device
        )

        # Vectorized calculations
        uv_base = uvw[:, :2]  # [n_vis, 2]
        cellsize_2pi = self.config.imaging_cellsize * 2 * np.pi

        # Processing by frequency (more memory efficient)
        for i, freq in enumerate(freqs):
            start_idx = i * n_vis
            end_idx = (i + 1) * n_vis

            # Vectorized calculation of normalized UV coordinates
            wavelength = const.c.value / freq
            uv_lambda = uv_base / wavelength
            uv_norm = (uv_lambda * cellsize_2pi).T

            samples_locs[:, start_idx:end_idx] = uv_norm
            all_visibilities[start_idx:end_idx] = visibilities[:, i]

        # Reshape for compatibility with rest of code
        visibilities_reshaped = all_visibilities.unsqueeze(0).unsqueeze(0)

        # Apply uniform weighting
        weights, valid_mask = self.uniform_weighting(
            samples_locs[0, :], samples_locs[1, :], im_size=im_size, device=self.device
        )

        # Filter out-of-bounds visibilities
        samples_locs = samples_locs[:, valid_mask]
        visibilities_reshaped = visibilities_reshaped[:, :, valid_mask]

        return samples_locs, weights, visibilities_reshaped

    @staticmethod
    def get_cellsize(sky_model, phase_center_ra, phase_center_dec, imaging_npixel):
        # Derive imaging grid directly from sky model extent to guarantee pixel alignment
        ra_values = sky_model[:, 0].to_numpy()
        dec_values = sky_model[:, 1].to_numpy()

        delta_ra = (ra_values - phase_center_ra) * np.cos(np.radians(phase_center_dec))
        delta_dec = dec_values - phase_center_dec
        max_ra_extent = np.max(np.abs(delta_ra))
        max_dec_extent = np.max(np.abs(delta_dec))
        half_fov_deg = max(max_ra_extent, max_dec_extent)
        fov_deg = 2.0 * half_fov_deg
        fov_rad = math.radians(fov_deg)
        imaging_cellsize = fov_rad / imaging_npixel

        return imaging_cellsize

    @staticmethod
    def uniform_weighting(
        u: torch.Tensor,
        v: torch.Tensor,
        im_size: torch.Tensor,
        weight_gridsize: int = 1,
        kernel_size: int = 5,
        device=torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized uniform weighting calculation for CPU/GPU


        device: torch.device
            Device to use for computations, it should match the self device.
        """

        dtype = torch.float32
        N0, N1 = [int(i * weight_gridsize) for i in im_size]

        # UV symmetrization (vectorized)
        flip_mask = v < 0
        u_sym = torch.where(flip_mask, -u, u)
        v_sym = torch.where(flip_mask, -v, v)

        # Grid indices calculation
        p = ((u_sym + 1) * N0 / 2).floor().to(torch.int64)
        q = ((v_sym + 1) * N1 / 2).floor().to(torch.int64)

        # Validity mask
        valid_mask = (p >= 0) & (p < N0) & (q >= 0) & (q < N1)
        p_valid = p[valid_mask]
        q_valid = q[valid_mask]

        uvInd = p_valid * N1 + q_valid

        # Weight calculation with optimized memory management
        gridded_weights = torch.zeros(N0 * N1, dtype=dtype, device=device)
        gridded_weights.scatter_add_(0, uvInd, torch.ones_like(uvInd, dtype=dtype))

        # Reshaping and filter application
        gridded_weights_2d = gridded_weights.view(1, 1, N0, N1)

        if device.type == "cuda":
            # Version optimisée GPU
            gridded_weights_2d = torch.nn.functional.avg_pool2d(
                gridded_weights_2d,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        else:
            # Version CPU - utilise des opérations plus efficaces
            kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=dtype) / (
                kernel_size * kernel_size
            )
            gridded_weights_2d = torch.nn.functional.conv2d(
                gridded_weights_2d, kernel, padding=kernel_size // 2
            )

        gridded_weights_2d = gridded_weights_2d.squeeze()
        gridded_weights_flat = gridded_weights_2d.contiguous().view(-1)

        # Calcul des poids finaux avec protection contre division par zéro
        weights = 1.0 / torch.clamp(gridded_weights_flat[uvInd], min=1e-8)

        return weights, valid_mask

    @staticmethod
    def display_uv_coverage(
        uv_normalized: torch.Tensor,
    ):
        """Display UV coverage"""

        plt.figure(figsize=(6, 6))
        plt.scatter(uv_normalized[0, :], uv_normalized[1, :], s=0.01)
        plt.xlabel("u (m)")
        plt.ylabel("v (m)")
        plt.title("Plan UV des visibilités")
        plt.grid()
        plt.axis("equal")
        plt.show()

    @staticmethod
    def bin_uv_data(
        uv_coords: torch.Tensor,
        visibilities: torch.Tensor,
        weights: torch.Tensor,
        grid_size: int = 512,
        device: torch.device = torch.device("cpu"),
    ):
        """Bin UV data to reduce number of visibilities


        Args:
        device: torch.device
            Device to use for computations, it should match the self device.
        """

        visibilities = visibilities.squeeze()  # [N]
        weights = weights.squeeze()  # [N]
        u, v = uv_coords[0], uv_coords[1]  # [N]

        u_norm = (u - u.min()) / (u.max() - u.min())
        v_norm = (v - v.min()) / (v.max() - v.min())

        u_idx = (u_norm * (grid_size - 1)).long()
        v_idx = (v_norm * (grid_size - 1)).long()
        idx = u_idx + v_idx * grid_size  # [N]

        max_bins = grid_size * grid_size

        sum_wu = torch.zeros(max_bins, device=device)
        sum_wv = torch.zeros(max_bins, device=device)
        sum_wvis_real = torch.zeros(max_bins, device=device)
        sum_wvis_imag = torch.zeros(max_bins, device=device)
        sum_w = torch.zeros(max_bins, device=device)

        sum_wu.index_add_(0, idx, u * weights)
        sum_wv.index_add_(0, idx, v * weights)
        sum_wvis_real.index_add_(0, idx, visibilities.real * weights)
        sum_wvis_imag.index_add_(0, idx, visibilities.imag * weights)
        sum_w.index_add_(0, idx, weights)

        mask = sum_w > 0

        u_binned = sum_wu[mask] / sum_w[mask]
        v_binned = sum_wv[mask] / sum_w[mask]

        vis_binned = (sum_wvis_real[mask] + 1j * sum_wvis_imag[mask]) / sum_w[mask]
        w_binned = sum_w[mask]

        binned_uv = torch.stack([u_binned, v_binned], dim=0)  # [2, M]
        vis_binned = vis_binned.unsqueeze(0).unsqueeze(0)  # [1,1,M]

        return binned_uv, w_binned, vis_binned

    def create_deepinv_physics(
        self,
        visibility_path: Path,
        visibility_format: str,
        visibility_column: str,
        bin_data: bool = False,
        imaging_npixel: Optional[int] = None,
        binning_factor: Optional[float] = None,
    ):
        # Load data
        uvw, visibilities, freqs = self.load_visibilities(
            visibility_path, visibility_format, visibility_column
        )
        # Normalize uv coords and compute weights
        samples_locs, weights, visibilities = self.normalize_uv_coords(
            uvw, freqs, visibilities
        )

        # Update default paramseters if provided
        imaging_npixel = (
            imaging_npixel if imaging_npixel is not None else self.config.imaging_npixel
        )
        binning_factor = (
            binning_factor if binning_factor is not None else self.config.binning_factor
        )

        if bin_data:
            samples_locs, weights, visibilities = self.bin_uv_data(
                samples_locs,
                visibilities,
                weights,
                grid_size=int(imaging_npixel * binning_factor),
                device=self.device,
            )

        physics = RadioInterferometry(
            img_size=torch.tensor([imaging_npixel, imaging_npixel]),
            samples_loc=samples_locs,
            real_projection=True,
            k_oversampling=self.config.nufft_k_oversampling,  # 2.0
            device=self.device,
        )

        physics.setWeight(weights)

        if self.verbose:
            print("visibilities", visibilities.shape)  # [N, channels, pol]
            print("samples_locs", samples_locs.shape)  # [2, N]
            print("weights", weights.shape)  # [channels, N]

        return physics, visibilities

    def create_psf(
        self,
        visibility_path: str,
        visibility_format: str = "MS",
        visibility_column: str = "DATA",
        bin_data: bool = False,
    ):
        if visibility_format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility_format} is not supported, "
                "currently only MS is supported for WSClean imaging"
            )

        physics, visibilities = self.create_deepinv_physics(
            visibility_path,
            visibility_format,
            visibility_column,
            bin_data=bin_data,
        )

        # Compute and normalize by PSF for calibrator
        psf = physics.A_adjoint(torch.ones_like(visibilities))

        if self.verbose:
            psf_peak = psf.max()
            print("psf_peak", psf_peak)

        return psf

    def create_dirty_image(
        self,
        visibility_path: str,
        visibility_format: str = "MS",
        visibility_column: str = "DATA",
        bin_data: bool = False,
    ):
        if visibility_format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility_format} is not supported, "
                "currently only MS is supported for WSClean imaging"
            )

        physics, visibilities = self.create_deepinv_physics(
            visibility_path,
            visibility_format,
            visibility_column,
            bin_data=bin_data,
        )

        back = physics.A_adjoint(visibilities)

        # Compute and normalize by PSF for calibrator
        psf = physics.A_adjoint(torch.ones_like(visibilities))
        psf_peak = psf.max()
        if self.verbose:
            print("psf_peak", psf_peak)

        back_normalized = back / psf_peak
        back_normalized = back_normalized.transpose(-2, -1).contiguous()
        # Flip
        back_normalized = back_normalized.flip(-2).contiguous()

        if self.verbose:
            print("Backprojection min value: ", back_normalized.min().item())
            print("Backprojection peak value: ", back_normalized.max().item())

        return back_normalized
