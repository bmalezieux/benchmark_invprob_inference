"""Benchmark different denoiser models with varying batch sizes.

This script:
1. Loads different denoiser models (DRUNet, UNet, DnCNN)
2. Tests a full image 2048x2048 pixels by processing it in patches of size 128, 256, and 512
3. Tests different batch sizes: 1, 2, 4, 8, 16
4. Tracks execution time and GPU memory usage
5. Generates interactive Plotly visualizations
"""
from __future__ import annotations

import torch
import sys
import os
# Add parent directory to path to allow importing benchmark_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from deepinv.models.drunet import DRUNet
from deepinv.models.unet import UNet
from deepinv.models.dncnn import DnCNN
from benchmark_utils.gpu_metrics import GPUMetricsTracker


def load_models(device: torch.device) -> Dict[str, torch.nn.Module]:
    """Load denoiser models.
    
    Parameters
    ----------
    device : torch.device
        Device to load models on
        
    Returns
    -------
    models : Dict[str, torch.nn.Module]
        Dictionary of loaded models
    """
    print(f"{'='*80}")
    print("Loading Models")
    print(f"{'='*80}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    models = {
        'DRUNet': DRUNet(in_channels=3, out_channels=3, pretrained='download').to(device),
        'UNet': UNet(in_channels=3, out_channels=3, scales=4, channels_per_scale=[64, 128, 256, 512], device=device),
        'DnCNN': DnCNN(in_channels=3, out_channels=3, pretrained='download').to(device),
    }
    
    for name, model in models.items():
        model.eval()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {num_params:,} parameters")
    
    return models


def benchmark_batch_size(model: torch.nn.Module, batch_sizes: List[int],
                        device: torch.device, model_name: str,
                        patch_size: int = 512, num_patches: int = 16,
                        num_runs: int = 3, num_warmup: int = 2) -> List[Dict]:
    """Benchmark denoiser model with varying batch sizes.
    
    Tests the model on different batch sizes with total of num_patches patches.
    Each patch has shape [batch_size, 3, patch_size, patch_size].
    
    Parameters
    ----------
    model : torch.nn.Module
        The denoiser model
    batch_sizes : List[int]
        List of batch sizes to test
    device : torch.device
        Device to run on
    model_name : str
        Name of the model
    patch_size : int
        Height and width of each patch (default: 512)
    num_patches : int
        Total number of patches to process (default: 16)
    num_runs : int
        Number of runs to average for each batch size
    num_warmup : int
        Number of warm-up runs

    Returns
    -------
    results : List[Dict]
        List of dictionaries with batch size, timing, memory and throughput metrics
    """
    print(f"\n{'='*80}")
    print(f"{model_name} Batch Size Benchmark (Patch: {patch_size}x{patch_size})")
    print(f"{'='*80}")
    print(f"Total patches: {num_patches}")
    print(f"Batch sizes: {batch_sizes}")
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n  Testing batch size: {batch_size}")
        
        # Create dummy patches
        # We'll process num_patches in chunks of batch_size
        num_batches = (num_patches + batch_size - 1) // batch_size  # Ceiling division
        
        # Warm-up steps
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        for _ in range(num_warmup):
            with torch.no_grad():
                # Process one batch for warmup
                current_batch_size = min(batch_size, num_patches)
                batch = torch.randn(current_batch_size, 3, patch_size, patch_size).to(device)
                _ = model(batch, sigma=0.1)
                del batch
        
        if device.type == 'cuda':
            torch.cuda.synchronize()

        tracker = GPUMetricsTracker(device)
        times = []
        memories = []
        throughputs = []
        
        for run in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            tracker.reset_iteration_tracking()
            
            with tracker.track_step(f'{model_name}_batch_{batch_size}'):
                with torch.no_grad():
                    # Process all patches in batches
                    for batch_idx in range(num_batches):
                        # Determine actual batch size for this iteration
                        # (last batch might be smaller)
                        current_batch_size = min(batch_size, num_patches - batch_idx * batch_size)
                        
                        # Create batch of patches
                        batch = torch.randn(current_batch_size, 3, patch_size, patch_size).to(device)
                        
                        # Forward pass
                        output = model(batch, sigma=0.1)
                        
                        del batch, output
            
            metrics = tracker.capture_iteration_result()
            step_key = f'{model_name}_batch_{batch_size}'
            run_time = metrics[f'{step_key}_time_sec']
            times.append(run_time)
            memories.append(metrics[f'{step_key}_memory_peak_mb'])
            throughputs.append(num_patches / run_time)

        # Calculate statistics
        avg_time_ms = np.mean(times) * 1000
        std_time_ms = np.std(times) * 1000
        avg_memory_mb = np.mean(memories)
        std_memory_mb = np.std(memories)
        
        # Calculate throughput stats
        avg_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)
        
        result = {
            'batch_size': batch_size,
            'avg_time_ms': avg_time_ms,
            'std_time_ms': std_time_ms,
            'avg_memory_mb': avg_memory_mb,
            'std_memory_mb': std_memory_mb,
            'avg_throughput': avg_throughput,
            'std_throughput': std_throughput,
            'total_pixels_processed': num_patches * patch_size * patch_size,
        }
        results.append(result)
        
        print(f"    Time: {avg_time_ms:.2f} ± {std_time_ms:.2f} ms")
        print(f"    Memory: {avg_memory_mb:.1f} ± {std_memory_mb:.1f} MB")
        print(f"    Throughput: {avg_throughput:.2f} ± {std_throughput:.2f} patches/sec")
    
    # Print summary table
    print(f"\n{'-'*130}")
    print(f"{model_name} Batch Size Summary:")
    print(f"{'Batch Size':<15} {'Time (ms)':<25} {'Memory (MB)':<25} {'Throughput (patches/s)':<25}")
    print(f"{'-'*130}")
    
    for result in results:
        print(f"{result['batch_size']:<15} {result['avg_time_ms']:.2f}±{result['std_time_ms']:.2f} ms       "
              f"{result['avg_memory_mb']:.1f}±{result['std_memory_mb']:.1f} MB     "
              f"{result['avg_throughput']:.2f}±{result['std_throughput']:.2f}")
    
    print(f"{'-'*130}\n")
    
    return results


def create_batch_size_plots(all_results_by_size: Dict[int, Dict[str, List[Dict]]]):
    """Create interactive Plotly visualizations of batch size benchmark results.
    
    Parameters
    ----------
    all_results_by_size : Dict[int, Dict[str, List[Dict]]]
        Dictionary mapping patch sizes to model results
    """
    
    models = list(list(all_results_by_size.values())[0].keys())
    patch_sizes = list(all_results_by_size.keys())
    colors = {'DRUNet': '#1f77b4', 'UNet': '#ff7f0e', 'DnCNN': '#2ca02c'}
    
    # -------------------------------------------------------------------------
    # 1. Peak Memory vs Batch Size (3 subplots for each patch size)
    # -------------------------------------------------------------------------
    fig_mem = make_subplots(
        rows=1, cols=len(patch_sizes),
        subplot_titles=[f"Patch Size {size}x{size}" for size in patch_sizes],
        horizontal_spacing=0.1
    )

    for i, patch_size in enumerate(patch_sizes):
        all_batch_results = all_results_by_size[patch_size]
        
        for model_name in models:
            results = all_batch_results[model_name]
            results_sorted = sorted(results, key=lambda x: x['batch_size'])
            
            batch_sizes_list = [r['batch_size'] for r in results_sorted]
            memories = [r['avg_memory_mb'] for r in results_sorted]
            memories_std = [r['std_memory_mb'] for r in results_sorted]

            # Only show legend for the first subplot
            show_legend = (i == 0)

            fig_mem.add_trace(
                go.Scatter(
                    x=batch_sizes_list,
                    y=memories,
                    error_y=dict(
                        type='data',
                        array=memories_std,
                        visible=True,
                        color=colors[model_name],
                    ),
                    mode='lines+markers',
                    name=f"{model_name}",
                    line=dict(color=colors[model_name], width=3),
                    marker=dict(size=10, symbol='diamond'),
                    legendgroup=model_name,
                    showlegend=show_legend,
                    hovertemplate='<b>%{fullData.name}</b><br>Batch Size: %{x}<br>Memory: %{y:.1f}±%{error_y.array:.1f}MB<extra></extra>',
                ),
                row=1, col=i+1
            )
            
            fig_mem.update_xaxes(title_text="Batch Size", row=1, col=i+1)
            fig_mem.update_yaxes(title_text="Peak Memory (MB)", row=1, col=i+1)

    fig_mem.update_layout(
        title="Peak Memory vs Batch Size",
        height=450,
        width=1200,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.3,
            xanchor='center',
            x=0.5
        )
    )
    fig_mem.write_html('batch_size_vs_memory.html')
    print("Saved batch_size_vs_memory.html")

    # -------------------------------------------------------------------------
    # 2. Throughput vs Batch Size (3 subplots for each patch size)
    # -------------------------------------------------------------------------
    fig_th = make_subplots(
        rows=1, cols=len(patch_sizes),
        subplot_titles=[f"Patch Size {size}x{size}" for size in patch_sizes],
        horizontal_spacing=0.15,
        specs=[[{"secondary_y": True}] * len(patch_sizes)]
    )

    for i, patch_size in enumerate(patch_sizes):
        all_batch_results = all_results_by_size[patch_size]
        patches_per_image = (2048 // patch_size) ** 2
        
        for model_name in models:
            results = all_batch_results[model_name]
            results_sorted = sorted(results, key=lambda x: x['batch_size'])
            
            batch_sizes_list = [r['batch_size'] for r in results_sorted]
            throughputs = [r['avg_throughput'] for r in results_sorted]
            throughputs_std = [r['std_throughput'] for r in results_sorted]
            
            # Calculate stats for secondary axis (Images/s)
            images_per_sec = [t / patches_per_image for t in throughputs]
            images_per_sec_std = [t / patches_per_image for t in throughputs_std]

            # Only show legend for the first subplot
            show_legend = (i == 0)

            # Define hover template with both Patches/s and Images/s
            hover_template = (
                '<b>%{fullData.name}</b><br>' +
                'Batch Size: %{x}<br>' +
                'Throughput: %{y:.2f} ± %{error_y.array:.2f} patches/s<br>' +
                'Images/s: %{customdata[0]:.2f} ± %{customdata[1]:.2f}<extra></extra>'
            )
            
            # Combine image stats for customdata
            custom_data = np.stack((images_per_sec, images_per_sec_std), axis=-1)

            # Primary Y-axis: Patches/s
            fig_th.add_trace(
                go.Scatter(
                    x=batch_sizes_list,
                    y=throughputs,
                    error_y=dict(
                        type='data',
                        array=throughputs_std,
                        visible=True,
                        color=colors[model_name]
                    ),
                    mode='lines+markers',
                    name=f"{model_name}",
                    line=dict(color=colors[model_name], width=3),
                    marker=dict(size=10, symbol='square'),
                    legendgroup=model_name,
                    showlegend=show_legend,
                    customdata=custom_data,
                    hovertemplate=hover_template,
                ),
                row=1, col=i+1, secondary_y=False
            )

            # Secondary Y-axis: Images/s (Invisible trace to set scale)
            fig_th.add_trace(
                go.Scatter(
                    x=batch_sizes_list,
                    y=images_per_sec,
                    mode='lines',
                    line=dict(width=0),
                    opacity=0,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=i+1, secondary_y=True
            )
            
            fig_th.update_xaxes(title_text="Batch Size", row=1, col=i+1)
            fig_th.update_yaxes(title_text="Throughput (patches/s)", row=1, col=i+1, secondary_y=False)
            fig_th.update_yaxes(title_text="Throughput (images/s)", row=1, col=i+1, secondary_y=True, showgrid=False)

    fig_th.update_layout(
        title="Throughput vs Batch Size",
        height=450,
        width=1200,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.3,
            xanchor='center',
            x=0.5
        )
    )
    fig_th.write_html('batch_size_vs_throughput.html')
    print("Saved batch_size_vs_throughput.html")

    return fig_mem, fig_th


def main():
    """Main execution function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_mem_gb:.2f} GB")
    else:
        print("WARNING: Running on CPU - this will be very slow.")
    
    # Load models
    models = load_models(device)
    
    # Configuration
    # We want to process roughly the equivalent of a 2048x2048 image
    # 2048x2048 = 4,194,304 pixels
    
    patch_sizes = [128, 256, 512]
    batch_sizes = [1, 2, 4, 8, 16]
    num_runs = 3
    num_warmup = 2
    
    # Calculate number of patches needed to cover 2048x2048 area
    # e.g., for 512x512, we need (2048/512)^2 = 16 patches
    num_patches_map = {
        128: (2048 // 128) ** 2,  # 256 patches
        256: (2048 // 256) ** 2,  # 64 patches
        512: (2048 // 512) ** 2,  # 16 patches
    }
    
    print(f"\n{'='*80}")
    print("Batch Size Benchmark Configuration")
    print(f"{'='*80}")
    print(f"Patch sizes: {patch_sizes}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Runs per batch size: {num_runs}")
    print(f"Equivalent image size: 2048x2048")
    
    # Benchmark each patch size
    all_results_by_size = {}
    
    for patch_size in patch_sizes:
        num_patches = num_patches_map[patch_size]
        print(f"\n{'='*80}")
        print(f"Testing Patch Size: {patch_size}x{patch_size} (Total patches: {num_patches})")
        print(f"{'='*80}")
        
        # Benchmark each model with varying batch sizes
        all_batch_results = {}
        for model_name, model in models.items():
            batch_results = benchmark_batch_size(
                model, batch_sizes, device, model_name,
                patch_size=patch_size, num_patches=num_patches, 
                num_runs=num_runs, num_warmup=num_warmup
            )
            all_batch_results[model_name] = batch_results
        
        all_results_by_size[patch_size] = all_batch_results
        
        # Print overall summary for this patch size
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR {patch_size}x{patch_size} PATCHES")
        print(f"{'='*80}")
        
        for model_name, results in all_batch_results.items():
            print(f"\n{model_name}:")
            print(f"{'Batch Size':<15} {'Time (ms)':<25} {'Memory (MB)':<25} {'Throughput (patches/s)':<25}")
            print(f"{'-'*90}")
            for result in results:
                print(f"{result['batch_size']:<15} {result['avg_time_ms']:.2f}±{result['std_time_ms']:.2f}        "
                      f"{result['avg_memory_mb']:.1f}±{result['std_memory_mb']:.1f}     "
                      f"{result['avg_throughput']:.2f}±{result['std_throughput']:.2f}")
        
        print(f"\n{'='*80}\n")
    
    # Create Plotly visualizations for all patch sizes
    print(f"\n{'='*80}")
    print(f"Creating Plotly Visualizations")
    print(f"{'='*80}")
    create_batch_size_plots(all_results_by_size)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETED")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
