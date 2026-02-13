import numpy as np
from pathlib import Path
from datetime import timedelta
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend

from benchmark_utils.radio_utils import get_meerkat_visibilities_path


def image_to_skymodel(image, ra_center, dec_center, pixel_size_deg):
    """
    Convert an image numpy array to a Karabo SkyModel.
    Each pixel becomes a point source.
    """
    if hasattr(image, "numpy"):
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

def image_to_skymodel_2(image_fits, ra_center, dec_center):

    data_fits = fits.open(image_fits)
    imaging_npixel = data_fits[0].data.shape[0]
    input_image_data = data_fits[0].data

    # Compute dynamic range of input image
    max_flux = np.max(input_image_data)
    # Calculate RMS excluding zero/NaN values
    valid_data = input_image_data[
        ~np.isnan(input_image_data) & (input_image_data != 0)
    ]
    if len(valid_data) > 0:
        # Computing the RMS using the lower 15% percentile to avoid bright sources
        # We assume that these pixels represent the noise background
        rms = np.std(valid_data[valid_data < np.quantile(valid_data, 0.15)])
        dynamic_range = max_flux / rms if rms > 0 else np.inf
    else:
        rms = 0
        dynamic_range = np.inf

    print(f"Input Image Dynamic Range for {image_fits}:")
    print(f"Max flux: {max_flux:.6e}")
    print(f"RMS: {rms:.6e}")
    print(
        f"  Dynamic Range: {dynamic_range:.2f} ({10*np.log10(dynamic_range):.2f} dB)"
    )

    data_fits.close()

    sky_model, _, _ = SkyModel.get_sky_model_from_optical_fits_image(
        str(image_fits),
        move_object=True,  # Default move to MeerKAT center
        new_ra=ra_center,
        new_dec=dec_center,
        # flux_percentile=0.0,
    )

    return sky_model, max_flux

def generate_meerkat_visibilities(
    fits_file,
    image: np.ndarray,
    cache_dir: Path,
    pixel_size_arcsec: float,
    integral_time: float = 7.997, # 7.997 sec integration
    use_gpus: bool = False,
    number_of_time_steps: int = 256,
    start_frequency_hz: float = 100e6,
    end_frequency_hz: float = 120e6,
    number_of_channels: int = 12
):
    """
    Generate visibilities for MeerKAT.
    Returns path to MS.
    """
    vis_path = get_meerkat_visibilities_path(
        image, cache_dir, pixel_size_arcsec, start_frequency_hz, number_of_time_steps, integral_time
    )
    
    if vis_path.exists():
        print(f"Loading cached visibilities from {vis_path}")
        return vis_path
        
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating new visibilities for MeerKAT in {vis_path}")
    
    # Setup MeerKAT
    telescope = Telescope.constructor("MeerKAT", backend=SimulatorBackend.OSKAR)
    
    # Setup Observation
    # Approx zenith at MeerKAT
    ra_center = 155.6
    dec_center = -30.7

    frequency_increment_hz = (end_frequency_hz - start_frequency_hz) / number_of_channels

    c = 299792458.0
    ref_freq = (start_frequency_hz + end_frequency_hz) / 2
    wavelength = c / ref_freq
    beam_fwhm_deg = np.degrees(1.2 * wavelength / 13)
    
    observation = Observation(
        mode="Tracking",
        phase_centre_ra_deg=ra_center,
        phase_centre_dec_deg=dec_center,
        start_frequency_hz=start_frequency_hz,
        frequency_increment_hz=frequency_increment_hz,
        number_of_channels=number_of_channels,
        start_date_and_time=Time("2020-04-26 16:36:00"),
        length=timedelta(seconds=number_of_time_steps * integral_time), 
        number_of_time_steps=number_of_time_steps
    )
    
    pixel_size_deg = pixel_size_arcsec / 3600.0
    #sky = image_to_skymodel(image, ra_center, dec_center, pixel_size_deg)
    sky, max_flux = image_to_skymodel_2(fits_file, ra_center, dec_center)

    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        pol_mode="Scalar", # Scalar = 1pol / Full = 4 pol
        station_type="Gaussian beam",
        gauss_beam_fwhm_deg=beam_fwhm_deg,
        gauss_ref_freq_hz=ref_freq,
        noise_enable=True,
        noise_start_freq=start_frequency_hz,
        noise_inc_freq=frequency_increment_hz,
        noise_number_freq=number_of_channels,
        noise_rms="Range",
        noise_rms_start=50,
        noise_rms_end=50,
        use_gpus=use_gpus
    )
    
    simulation.run_simulation(
        telescope,
        sky,
        observation,
        visibility_format="MS",
        visibility_path=str(vis_path)
    )
    
    return vis_path
