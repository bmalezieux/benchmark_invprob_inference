import hashlib
import yaml
import json
import types
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.coordinates import (
    SkyCoord,
    EarthLocation,
    AltAz,
    Longitude,
    Angle,
    Latitude,
    ICRS,
    FK5,
)
from astropy.time import Time
import astropy.units as u
from scipy.ndimage import zoom

MEERKAT_LOCATION = EarthLocation(
    lat=-30.83 * u.deg, lon=21.33 * u.deg, height=1195.0 * u.m
)

def load_and_resize_image(image_path, image_size):
    """Load and resize m1_n.fits image.
    
    Returns
    -------
    np.ndarray
        The resized image of shape (C, H, W) in [0, 1].
    """
    with fits.open(image_path, memmap=False) as hdul:
        img = np.array(hdul[0].data, dtype=np.float32, copy=True)

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(img))

    # Normalize to [0, 1]
    if max_val > 1.0:
        img = img / max_val

    img = np.squeeze(img)

    # Ensure (C, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim != 3:
        raise ValueError(
            f"Unexpected FITS image shape {img.shape}, expected 2D or 3D after squeeze."
        )
    
    c, h, w = img.shape
    resized_img = img.copy()
    
    if h != image_size or w != image_size:
        zoom_factors = (1, image_size / h, image_size / w)
        resized_img = zoom(img, zoom_factors, order=3)
        resized_img = np.clip(resized_img, 0, 1)

    return np.ascontiguousarray(resized_img, dtype=np.float32)

'''def get_meerkat_visibilities_path(
    image: np.ndarray,
    cache_dir: Path,
    start_frequency_hz: float = 1e9,
    number_of_time_steps: int = 256,
    integral_time: float = 10, # 10 sec integration
):
    """
    Generate path for MeerKAT visibilities.
    """
    # Create a unique hash for the simulation parameters
    params = {
        'start_frequency_hz': start_frequency_hz,
        'number_of_time_steps': number_of_time_steps,
        'integral_time': integral_time
    }
    params_str = str(sorted(params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()

    if hasattr(image, "cpu") and hasattr(image, "numpy"):
        img_bytes = image.cpu().numpy().tobytes()
    else:
        img_bytes = image.tobytes()

    img_hash = hashlib.md5(img_bytes).hexdigest()
    full_hash = hashlib.md5((params_hash + img_hash).encode()).hexdigest()

    vis_path = cache_dir / f"{full_hash}.ms"
    return vis_path'''

def get_meerkat_visibilities_path(
    image: np.ndarray,
    cache_dir: Path,
    fits_name: str
):
    """
    Generate path for MeerKAT visibilities.
    """
    # Create a unique hash for the simulation parameters
    params = {
        'fits_name': fits_name
    }
    params_str = str(sorted(params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()

    if hasattr(image, "cpu") and hasattr(image, "numpy"):
        img_bytes = image.cpu().numpy().tobytes()
    else:
        img_bytes = image.tobytes()

    img_hash = hashlib.md5(img_bytes).hexdigest()
    full_hash = hashlib.md5((params_hash + img_hash).encode()).hexdigest()

    vis_path = cache_dir / f"{full_hash}.ms"
    return vis_path

def load_object(dct):
    return types.SimpleNamespace(**dct)

def load_config(config_path, section=None):
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    if section is not None:
        if not hasattr(cfg, section):
            raise KeyError(f"Section '{section}' not found in config: {config_path}")
        return getattr(cfg, section)

    return cfg

def is_source_visible(
    ra_deg,
    dec_deg,
    obs_start_time,
    obs_duration,
    telescope_location,
    min_elevation_deg=15.0,
    n_time_samples=10,
):
    """
    Check if a source is visible (above horizon) for the entire observation duration.

    Args:
        ra_deg: Right Ascension in degrees
        dec_deg: Declination in degrees
        obs_start_time: Observation start time (datetime object)
        obs_duration: Observation duration (timedelta object)
        telescope_location: EarthLocation of the telescope
        min_elevation_deg: Minimum elevation above horizon in degrees (default: 15)
        n_time_samples: Number of time samples to check during observation

    Returns:
        bool: True if source is visible for entire observation, False otherwise
    """
    # Create sky coordinate
    source = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    # Sample times throughout the observation
    time_samples = [
        obs_start_time + i * obs_duration / (n_time_samples - 1)
        for i in range(n_time_samples)
    ]

    # Check elevation at each time sample
    for t in time_samples:
        obs_time = Time(t)
        altaz_frame = AltAz(obstime=obs_time, location=telescope_location)
        source_altaz = source.transform_to(altaz_frame)

        if source_altaz.alt.deg < min_elevation_deg:
            return False

    return True

def draw_random_pointing(
    time: Time,
    observer: EarthLocation = MEERKAT_LOCATION,
    min_elevation_deg: float = 15.0,
    max_attempts: int = 1000,
    n_azimuth_samples: int = 360,
) -> tuple:
    """Draw a random pointing (RA/Dec) inside the region defined by a minimum elevation contour.

    This function computes the elevation contour at a given time and observer location,
    then randomly samples a point within that region.

    Args:
        time: Time at which to draw the sample
        observer: Earth location of the observer
        min_elevation_deg: Minimum elevation in degrees defining the contour
        max_attempts: Maximum number of random attempts to find a valid point
        n_azimuth_samples: Number of azimuth samples to define the contour boundary

    Returns:
        tuple: (ra_deg, dec_deg) coordinates of the random pointing in degrees

    Raises:
        RuntimeError: If no valid pointing is found after max_attempts

    Example:
        >>> from astropy.time import Time
        >>> from astropy.coordinates import EarthLocation
        >>> import astropy.units as u
        >>>
        >>> time = Time("2020-04-26T16:36:00")
        >>> meerkat = EarthLocation(lat=-30.83*u.deg, lon=21.33*u.deg, height=1195.0*u.m)
        >>> ra, dec = draw_random_pointing_in_elevation_contour(
        ...     time=time,
        ...     observer=meerkat,
        ...     min_elevation_deg=15.0
        ... )
    """
    # Sample azimuth angles to define the elevation contour boundary
    azimuth_sample = np.linspace(0, 360, n_azimuth_samples)
    elevation_boundary = np.full(azimuth_sample.size, min_elevation_deg)

    # Convert boundary to RA/Dec
    boundary_altaz = SkyCoord(
        azimuth_sample * u.deg,
        elevation_boundary * u.deg,
        frame=AltAz(obstime=time, location=observer),
    )
    boundary_radec = boundary_altaz.transform_to(ICRS)

    # Get RA/Dec ranges from the boundary
    ra_deg = boundary_radec.ra.deg
    dec_deg = boundary_radec.dec.deg

    # Compute approximate bounds (accounting for RA wrapping)
    dec_min = np.min(dec_deg)
    dec_max = np.max(dec_deg)

    # For RA, handle potential wrapping around 0/360
    ra_range = np.ptp(ra_deg)  # peak-to-peak (max - min)
    if ra_range > 180:  # Wrapping detected
        # Use the full RA range
        ra_min = 0
        ra_max = 360
    else:
        ra_min = np.min(ra_deg)
        ra_max = np.max(ra_deg)

    # Randomly sample points within the bounding box
    for attempt in range(max_attempts):
        # Generate random RA/Dec within bounds
        ra_candidate = ra_min + np.random.rand() * (ra_max - ra_min)
        dec_candidate = dec_min + np.random.rand() * (dec_max - dec_min)

        # Convert to AltAz to check elevation
        candidate_radec = SkyCoord(
            ra_candidate * u.deg, dec_candidate * u.deg, frame=ICRS
        )
        candidate_altaz = candidate_radec.transform_to(
            AltAz(obstime=time, location=observer)
        )

        # Check if elevation meets the minimum requirement
        if candidate_altaz.alt.deg >= min_elevation_deg:
            return ra_candidate, dec_candidate

    raise RuntimeError(
        f"Could not find valid pointing within elevation contour after {max_attempts} attempts. "
        f"Try increasing max_attempts or reducing min_elevation_deg."
    )
