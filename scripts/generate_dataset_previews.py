#!/usr/bin/env python
"""
Generate preview images for all datasets in the benchmark.

This script creates visualization previews for:
- High-resolution color images
- 2D tomography (Shepp-Logan phantom)
- 3D tomography (Walnut cone-beam CT)

All previews are saved under docs/source/_static/images/ for documentation.

Usage:
    python scripts/generate_dataset_previews.py [dataset_name]
    
    If dataset_name is provided, only generate that preview.
    Otherwise, generate all previews.
    
    Valid dataset names: highres_color_image, tomography_2d, tomography_3d
"""

import sys
import shutil
from pathlib import Path
import importlib

from benchopt import config as bench_config
from benchopt.config import yaml
from deepinv.utils.plotting import plot, plot_ortho3D


# Resolve project root and add to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# Dataset configuration mapping
DATASET_CONFIGS = {
    'highres_color_image': {
        'config_file': 'highres_imaging.yml',
        'config_key': 'highres_color_image',
        'output_file': 'highres_preview.png',
        'module': 'datasets.highres_color_image',
        'plot_type': '2d',
        'title': 'Ground Truth',
        'measurement_titles': ['Measurement 1', 'Measurement 2'],
    },
    'tomography_2d': {
        'config_file': 'tomography_2d.yml',
        'config_key': 'tomography_2d',
        'output_file': 'tomography_2d_preview.png',
        'module': 'datasets.tomography_2d',
        'plot_type': '2d',
        'title': 'Ground Truth (Shepp-Logan)',
    },
    'tomography_3d': {
        'config_file': 'tomography_3d.yml',
        'config_key': 'tomography_3d',
        'output_file': 'tomography_3d_preview.png',
        'module': 'datasets.tomography_3d',
        'plot_type': '3d',
        'title': 'Ground Truth',
    },
}


def generate_preview(dataset_name):
    """Generate preview for a specific dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Valid options: {list(DATASET_CONFIGS.keys())}")
    
    cfg = DATASET_CONFIGS[dataset_name]
    print(f"\n=== Generating {dataset_name} Preview ===")
    
    # Import dataset module and load config
    Dataset = importlib.import_module(cfg['module']).Dataset
    config_path = project_root / 'configs' / cfg['config_file']
    with open(config_path, 'r') as f:
        config_content = yaml.safe_load(f)
    dataset_params = config_content['dataset'][0][cfg['config_key']]
    
    # Setup and load dataset
    bench_config.get_data_path = lambda key=None: str(project_root / "data")
    ds = Dataset(**dataset_params, seed=42)
    data = ds.get_data()
    
    # Setup output
    out_dir = project_root / 'docs' / 'source' / '_static' / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg['output_file']
    
    ground_truth = data['ground_truth']
    print(f"Ground truth shape: {ground_truth.shape}")
    
    if cfg['plot_type'] == '3d':
        # 3D tomography: use orthogonal view
        plot_ortho3D(
            ground_truth.detach().cpu(),
            titles=[cfg['title']],
            save_dir=out_dir / "temp_3d",
            tight=True,
            show=False,
            fontsize=8,
        )
        temp_img = out_dir / "temp_3d" / "images.png"
        if temp_img.exists():
            shutil.copy(temp_img, out_path)
            shutil.rmtree(out_dir / "temp_3d")
    else:
        plot_list = [ground_truth]
        titles = [cfg['title']]
        measurements = data['measurement']
        
        if 'measurement_titles' in cfg:
            for i, title in enumerate(cfg['measurement_titles']):
                plot_list.append(measurements[i])
                titles.append(title)
        else:
            for i, meas in enumerate(measurements):
                plot_list.append(meas)
                titles.append(f'Measurement {i+1}')
        
        plot(plot_list, titles=titles, save_fn=out_path, tight=True, show=False,fontsize=8)
    
    print(f"Saved: {out_path}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        generate_preview(sys.argv[1].lower())
    else:
        print("Generating all dataset previews...")
        for dataset_name in DATASET_CONFIGS.keys():
            generate_preview(dataset_name)
        print("\n=== All previews generated successfully ===")


if __name__ == "__main__":
    main()
