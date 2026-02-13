import hashlib
import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import zoom

def load_and_resize_image(image_path, image_size):
    """Load and resize m1_n.fits image.
    
    Returns
    -------
    np.ndarray
        The resized image of shape (C, H, W) in [0, 1].
    """
    with fits.open(image_path) as hdul:
        img = hdul[0].data.astype(np.float32)

        # Normalize to [0, 1]
        if img.max() > 1.0:
            img /= img.max()

    # Ensure (C, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    
    c, h, w = img.shape
    resized_img = img.copy()
    
    if h != image_size or w != image_size:
        zoom_factors = (1, image_size / h, image_size / w)
        resized_img = zoom(img, zoom_factors, order=3)
        resized_img = np.clip(resized_img, 0, 1)
        
    return resized_img

def get_meerkat_visibilities_path(
    image: np.ndarray,
    cache_dir: Path,
    pixel_size_arcsec: float,
    start_frequency_hz: float = 1e9,
    number_of_time_steps: int = 256,
    integral_time: float = 10, # 10 sec integration
):
    """
    Generate path for MeerKAT visibilities.
    """
    # Create a unique hash for the simulation parameters
    params = {
        'pixel_size_arcsec': pixel_size_arcsec,
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
    return vis_path
