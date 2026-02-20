import numpy as np
import os
import math
from pathlib import Path
from datetime import timedelta, timezone, datetime
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel, get_cellsize
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend

from benchmark_utils.radio_utils import get_meerkat_visibilities_path, load_config, MEERKAT_LOCATION, is_source_visible, draw_random_pointing

def set_phase_center(pos_ra, pos_dec, random_position, number_of_time_steps, min_elevation=15.0, verbose=0, sim_it=0):
    verbose = 0
    sim_it = 0
    n_simulations = 1
    # Observation data and time
    # zenith at MeerKAT around 18:36 UTC with RA=155.66367, Dec=-30.7130
    obs_date_time = datetime(2020, 4, 26, 16, 36, 0, 0, timezone.utc)

    if random_position:
        try:
            # Convert obs_date_time to astropy Time
            approx_duration = timedelta(seconds=number_of_time_steps * 7.997)

            # Retry loop to ensure visible pointing
            max_retries = 10
            for attempt in range(max_retries):

                phase_center_ra, phase_center_dec = draw_random_pointing(
                    time=Time(obs_date_time),
                    observer=MEERKAT_LOCATION,
                    min_elevation_deg=min_elevation,
                )

                # Verify visibility for all the observation duration
                if is_source_visible(
                    ra_deg=phase_center_ra,
                    dec_deg=phase_center_dec,
                    obs_start_time=obs_date_time,
                    obs_duration=approx_duration,
                    telescope_location=MEERKAT_LOCATION,
                    min_elevation_deg=min_elevation,
                ):
                    break
                elif attempt == max_retries - 1:
                    print(
                        f"Warning: Could not find visible pointing after {max_retries} attempts"
                    )
                    print(
                        "WARNING: Selected position may go below horizon during observation!"
                    )
            if verbose:
                print("\n" + "=" * 60)
                print(f"Random Pointing {sim_it + 1}/{n_simulations}:")
                print("=" * 60)
                print(
                    f"  RA  = {phase_center_ra:8.4f}° ({phase_center_ra/15:8.4f} hours)"
                )
                print(f"  DEC = {phase_center_dec:8.4f}°")
                print("=" * 60 + "\n")

        except Exception as e:
            print(f"Warning: RandomPointing failed: {e}")
            print("Falling back to non-random position")
            phase_center_ra = pos_ra
            phase_center_dec = pos_dec
    else:
        phase_center_ra = pos_ra
        phase_center_dec = pos_dec
        if verbose:
            print("\n" + "=" * 60)
            print(f"Fixed Pointing {sim_it + 1}/{n_simulations}:")
            print("=" * 60)
            print(
                f"  RA  = {phase_center_ra:8.4f}° ({phase_center_ra/15:8.4f} hours)"
            )
            print(f"  DEC = {phase_center_dec:8.4f}°")
            print("=" * 60 + "\n")

    return phase_center_ra, phase_center_dec, obs_date_time

def image_to_skymodel(image_fits, ra_center, dec_center):

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

    return sky_model, max_flux, imaging_npixel

def generate_meerkat_visibilities(
    fits_file,
    image: np.ndarray,
    cache_dir: Path,
    use_gpus: bool = False,
    number_of_time_steps: int = 256,
    start_frequency_hz: float = 100e6,
    end_frequency_hz: float = 120e6,
    number_of_channels: int = 12,
    pos_ra: float = 155.66367,
    pos_dec: float = -30.7130,
    random_position: bool = False,
    add_noise: bool = False
):
    """
    Generate visibilities for MeerKAT.
    Returns path to MS.
    """
    vis_path = get_meerkat_visibilities_path(
        image, cache_dir, os.path.basename(fits_file)
    )
    
    if vis_path.exists():
        print(f"Loading cached visibilities from {vis_path}")
        return vis_path
        
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating new visibilities for MeerKAT in {vis_path}")

    phase_center_ra, phase_center_dec, obs_date_time = set_phase_center(pos_ra, pos_dec, random_position, number_of_time_steps)

    sky, max_flux, imaging_npixel = image_to_skymodel(fits_file, phase_center_ra, phase_center_dec)

    cellsize = get_cellsize(sky, phase_center_ra, phase_center_dec, imaging_npixel)
    
    # Setup MeerKAT
    telescope = Telescope.constructor("MeerKAT", backend=SimulatorBackend.OSKAR)

    # From survey metadata
    frequency_increment_hz = math.floor(
        (end_frequency_hz - start_frequency_hz) / number_of_channels
    )

    # number_of_channels = 1
    print(f"number_of_channels={number_of_channels}")

    c = 299792458.0
    ref_freq = (start_frequency_hz + end_frequency_hz) / 2
    wavelength = c / ref_freq
    beam_fwhm_deg = np.degrees(1.2 * wavelength / 13)
    
    # Define observation
    observation = Observation(
        phase_centre_ra_deg=phase_center_ra,
        phase_centre_dec_deg=phase_center_dec,
        start_date_and_time=obs_date_time,
        length=timedelta(seconds=number_of_time_steps * 7.997),
        number_of_time_steps=number_of_time_steps,
        number_of_channels=number_of_channels,
        start_frequency_hz=start_frequency_hz,
        frequency_increment_hz=frequency_increment_hz,
    )

    if add_noise:
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=frequency_increment_hz,
            pol_mode="Scalar",  # Scalar = 1pol / Full = 4 pol
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
            use_gpus=use_gpus,
        )
    else:
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=frequency_increment_hz,
            pol_mode="Scalar",  # Scalar = 1pol / Full = 4 pol
            station_type="Gaussian beam",
            gauss_beam_fwhm_deg=beam_fwhm_deg,
            gauss_ref_freq_hz=ref_freq,
            noise_enable=False,
            use_gpus=use_gpus,
        )
    
    simulation.run_simulation(
        telescope,
        sky,
        observation,
        visibility_format="MS",
        visibility_path=str(vis_path)
    )
    
    return vis_path
