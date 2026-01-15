import os
import glob
import hashlib
import numpy as np
import torch
from pathlib import Path
from astropy.time import Time
from astropy import units as u
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend

def image_to_skymodel(image, ra_center, dec_center, pixel_size_deg):
    """
    Convert an image tensor to a Karabo SkyModel.
    Each pixel becomes a point source.
    """
    if torch.is_tensor(image):
        image = image.numpy()
        
    if image.ndim == 3:
        image = image[0] # Take first channel if (C, H, W)
        
    h, w = image.shape
    
    # Create grid of RA/Dec
    # RA increases to the left (East)
    x = np.arange(w) - w/2.0 + 0.5
    y = np.arange(h) - h/2.0 + 0.5
    
    # grid in degrees
    x_deg = x * pixel_size_deg
    y_deg = y * pixel_size_deg
    
    # Convert to RA/Dec
    # approx mapping centered at (ra_center, dec_center)
    ra_grid = ra_center + x_deg / np.cos(np.deg2rad(dec_center))
    dec_grid = dec_center + y_deg
    
    X, Y = np.meshgrid(ra_grid, dec_grid)
    
    # Filter out zero pixels to save computation
    mask = image > 1e-9
    
    # SkyModel expects [ra, dec, stokes_I, ...]
    ra_flat = X[mask]
    dec_flat = Y[mask]
    flux_flat = image[mask]
    
    sky = SkyModel()
    
    num_sources = len(flux_flat)
    data = np.zeros((num_sources, 12))
    data[:, 0] = ra_flat
    data[:, 1] = dec_flat
    data[:, 2] = flux_flat # Stokes I
    data[:, 6] = 100e6     # ref_freq (dummy)
    
    sky.add_point_sources(data)
    
    return sky

def get_meerkat_visibilities(
    image: torch.Tensor,
    cache_dir: Path,
    pixel_size_arcsec: float,
    freq_hz: float = 1e9,
    obs_duration: float = 600, # 10 min
    integral_time: float = 10, # 10 sec integration
    device: str = "cpu",
):
    """
    Generate or load visibilities for MeerKAT.
    Returns path to MS.
    """
    # Create a unique hash for the simulation parameters
    params = {
        'pixel_size_arcsec': pixel_size_arcsec,
        'freq_hz': freq_hz,
        'obs_duration': obs_duration,
        'integral_time': integral_time
    }
    params_str = str(sorted(params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()

    if torch.is_tensor(image):
        img_bytes = image.cpu().numpy().tobytes()
    else:
        img_bytes = image.tobytes()

    img_hash = hashlib.md5(img_bytes).hexdigest()
    full_hash = hashlib.md5((params_hash + img_hash).encode()).hexdigest()

    vis_path = cache_dir / f"{full_hash}.ms"
    
    if vis_path.exists():
        print(f"Loading cached visibilities from {vis_path}")
        return vis_path
        
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating new visibilities for MeerKAT in {vis_path}")
    
    # Setup MeerKAT
    telescope = Telescope.constructor("MeerKAT", backend=SimulatorBackend.OSKAR)
    
    # Setup Observation
    # Approx zenith at MeerKAT
    ra_center = 20.0
    dec_center = -30.0
    
    observation = Observation(
        mode="Tracking",
        pointing_ra=ra_center,
        pointing_dec=dec_center,
        phase_center_ra=ra_center,
        phase_center_dec=dec_center,
        start_frequency_hz=freq_hz,
        frequency_increment_hz=1e6,
        number_of_channels=1,
        start_date_and_time=Time("2020-04-26 16:36:00"),
        length=Time(obs_duration * u.s).to_datetime(), 
        number_of_time_steps=int(obs_duration / integral_time)
    )
    
    pixel_size_deg = pixel_size_arcsec / 3600.0
    sky = image_to_skymodel(image, ra_center, dec_center, pixel_size_deg)
    
    # Configure simulation
    use_gpus = False
    if device != "cpu":
        use_gpus = True

    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        use_gpus=use_gpus
    )
    
    simulation.run_simulation(
        telescope,
        sky,
        observation,
        visibility_format="MS",
        visibility_path=vis_path
    )
    
    return vis_path
