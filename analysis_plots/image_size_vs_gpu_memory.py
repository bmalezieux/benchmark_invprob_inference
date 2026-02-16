"""Benchmark different denoiser models with varying image sizes

This script:
1. Loads different denoiser models (DRUNet, UNet, DnCNN)
2. Varying image size: 1024 to 2048 (width) with fixed height=2048
3. Performs forward passes for each model on each image
4. Prints execution time and memory usage
5. Generates interactive Plotly visualizations
"""

from __future__ import annotations

import os
import sys

import torch

# Add parent directory to path to allow importing benchmark_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import statistics
from typing import Dict, List

import plotly.graph_objects as go
from deepinv.models.dncnn import DnCNN
from deepinv.models.drunet import DRUNet
from deepinv.models.unet import UNet
from plotly.subplots import make_subplots

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

    if device.type == "cuda":
        torch.cuda.empty_cache()

    models = {
        "DRUNet": DRUNet(in_channels=3, out_channels=3, pretrained="download").to(
            device
        ),
        "UNet": UNet(
            in_channels=3,
            out_channels=3,
            scales=4,
            channels_per_scale=[64, 128, 256, 512],
            device=device,
        ),
        "DnCNN": DnCNN(in_channels=3, out_channels=3, pretrained="download").to(device),
    }

    for name, model in models.items():
        model.eval()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {num_params:,} parameters")

    return models


def benchmark_image_size(
    model: torch.nn.Module,
    device: torch.device,
    model_name: str,
    step: int,
    num_runs: int = 3,
    num_warmup: int = 3,
):
    """Benchmark denoiser model with varying image sizes.

    Tests the model on images with height=2048 and varying widths.

    Parameters
    ----------
    model : torch.nn.Module
        The denoiser model
    device : torch.device
        Device to run on
    model_name : str
        Name of the model
    step : int
        Step size for image width
    num_runs : int
        Number of runs to average
    num_warmup : int
        Number of warm-up runs
    """
    print(f"\n{'='*80}")
    print(f"{model_name} Scaling Test: Varying Image Widths (Step {step})")
    print(f"{'='*80}")

    # Define image sizes
    image_sizes = [(2048, width) for width in range(1024, 2049, step)]

    scaling_results = []

    for height, width in image_sizes:
        image_name = f"{height}x{width}"
        print(f"\n  Testing image size: {image_name}")

        # Generate image
        image = torch.randn(1, 3, height, width)
        image_device = image.to(device)

        # Warm-up steps
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(image_device, sigma=0.1)
        if device.type == "cuda":
            torch.cuda.synchronize()

        tracker = GPUMetricsTracker(device)
        times = []
        memories = []

        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            tracker.reset_iteration_tracking()

            with tracker.track_step(f"{model_name}_{image_name}"):
                with torch.no_grad():
                    output = model(image_device, sigma=0.1)

            metrics = tracker.capture_iteration_result()
            step_key = f"{model_name}_{image_name}"
            times.append(metrics[f"{step_key}_time_sec"])
            memories.append(metrics[f"{step_key}_memory_peak_mb"])

        # Get final metrics
        avg_time_ms = (sum(times) / len(times)) * 1000
        std_time_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0

        avg_memory_mb = sum(memories) / len(memories)
        std_memory_mb = statistics.stdev(memories) if len(memories) > 1 else 0.0

        result = {
            "image_size": image_name,
            "height": height,
            "width": width,
            "num_pixels": height * width,
            "avg_time_ms": avg_time_ms,
            "std_time_ms": std_time_ms,
            "avg_memory_mb": avg_memory_mb,
            "std_memory_mb": std_memory_mb,
            "throughput_mp_per_sec": (height * width) / (avg_time_ms / 1000) / 1e6,
        }
        scaling_results.append(result)

        print(f"    Time: {avg_time_ms:.2f} \u00b1 {std_time_ms:.2f} ms")
        print(f"    Memory: {avg_memory_mb:.1f} \u00b1 {std_memory_mb:.1f} MB")
        print(f"    Throughput: {result['throughput_mp_per_sec']:.2f} MP/s")

        del image, image_device, output

    # Print summary table
    print(f"\n{'-'*80}")
    print(f"{model_name} Scaling Summary (Step {step}):")
    print(
        f"{'Image Size':<15} {'Pixels':<15} {'Time (ms)':<25} {'Memory (MB)':<25} {'Throughput (MP/s)':<18}"
    )
    print(f"{'-'*98}")

    for result in scaling_results:
        time_str = f"{result['avg_time_ms']:.2f} \u00b1 {result['std_time_ms']:.2f}"
        mem_str = f"{result['avg_memory_mb']:.1f} \u00b1 {result['std_memory_mb']:.1f}"
        print(
            f"{result['image_size']:<15} {result['num_pixels']:<15,d} "
            f"{time_str:<25} {mem_str:<25} "
            f"{result['throughput_mp_per_sec']:<18.2f}"
        )

    print(f"{'-'*98}\n")

    return scaling_results


def create_scaling_plots(
    all_scaling_results_64: Dict[str, List[Dict]],
    all_scaling_results_68: Dict[str, List[Dict]],
):
    """Create interactive Plotly visualizations of scaling test results.

    Parameters
    ----------
    all_scaling_results_64 : Dict[str, List[Dict]]
        Dictionary mapping model names to their scaling results (Step 64)
    all_scaling_results_68 : Dict[str, List[Dict]]
        Dictionary mapping model names to their scaling results (Step 68)
    """
    print(f"\n{'='*80}")
    print("Creating Plotly Visualizations")
    print(f"{'='*80}")

    models = list(all_scaling_results_64.keys())
    colors = {"DRUNet": "#1f77b4", "UNet": "#ff7f0e", "DnCNN": "#2ca02c"}

    # -------------------------------------------------------------------------
    # 1. Time vs Image Size Plot
    # -------------------------------------------------------------------------
    # Determine common y-axis range for time
    all_times = []
    for results in all_scaling_results_64.values():
        all_times.extend([r["avg_time_ms"] for r in results])
    for results in all_scaling_results_68.values():
        all_times.extend([r["avg_time_ms"] for r in results])

    # Add a small buffer (5%)
    min_time = min(all_times) * 0.95
    max_time = max(all_times) * 1.05

    fig_time = make_subplots(
        rows=1, cols=2, subplot_titles=("Step 64", "Step 68"), horizontal_spacing=0.1
    )

    for model_name in models:
        # Step 64
        results_64 = all_scaling_results_64[model_name]
        results_sorted_64 = sorted(results_64, key=lambda x: x["num_pixels"])
        widths_64 = [r["width"] for r in results_sorted_64]
        times_64 = [r["avg_time_ms"] for r in results_sorted_64]
        std_times_64 = [r["std_time_ms"] for r in results_sorted_64]

        # Step 68
        results_68 = all_scaling_results_68[model_name]
        results_sorted_68 = sorted(results_68, key=lambda x: x["num_pixels"])
        widths_68 = [r["width"] for r in results_sorted_68]
        times_68 = [r["avg_time_ms"] for r in results_sorted_68]
        std_times_68 = [r["std_time_ms"] for r in results_sorted_68]

        # Add traces for Step 64 (Left)
        fig_time.add_trace(
            go.Scatter(
                x=widths_64,
                y=times_64,
                error_y=dict(type="data", array=std_times_64, visible=True),
                mode="lines+markers",
                name=f"{model_name}",
                legendgroup=model_name,
                line=dict(color=colors[model_name], width=3),
                marker=dict(size=8, symbol="circle"),
                hovertemplate="<b>%{fullData.name}</b><br>Width: %{x}<br>Time: %{y:.2f} ± %{error_y.array:.2f}ms<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add traces for Step 68 (Right)
        fig_time.add_trace(
            go.Scatter(
                x=widths_68,
                y=times_68,
                error_y=dict(type="data", array=std_times_68, visible=True),
                mode="lines+markers",
                name=f"{model_name}",
                legendgroup=model_name,
                showlegend=False,
                line=dict(color=colors[model_name], width=3),
                marker=dict(size=8, symbol="diamond"),
                hovertemplate="<b>%{fullData.name}</b><br>Width: %{x}<br>Time: %{y:.2f} ± %{error_y.array:.2f}ms<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig_time.update_layout(
        title="Execution Time vs Image Width",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
    )
    fig_time.update_xaxes(title_text="Image Width (pixels)", row=1, col=1)
    fig_time.update_yaxes(
        title_text="Time (ms)", range=[min_time, max_time], row=1, col=1
    )
    fig_time.update_xaxes(title_text="Image Width (pixels)", row=1, col=2)
    fig_time.update_yaxes(
        title_text="Time (ms)", range=[min_time, max_time], row=1, col=2
    )

    fig_time.write_html("image_size_vs_time.html")
    print("Saved image_size_vs_time.html")

    # -------------------------------------------------------------------------
    # 2. Memory vs Image Size Plot
    # -------------------------------------------------------------------------
    # Determine common y-axis range for memory
    all_mems = []
    for results in all_scaling_results_64.values():
        all_mems.extend([r["avg_memory_mb"] for r in results])
    for results in all_scaling_results_68.values():
        all_mems.extend([r["avg_memory_mb"] for r in results])

    # Add a small buffer (5%)
    min_mem = min(all_mems) * 0.95
    max_mem = max(all_mems) * 1.05

    fig_mem = make_subplots(
        rows=1, cols=2, subplot_titles=("Step 64", "Step 68"), horizontal_spacing=0.1
    )

    for model_name in models:
        # Step 64
        results_64 = all_scaling_results_64[model_name]
        results_sorted_64 = sorted(results_64, key=lambda x: x["num_pixels"])
        widths_64 = [r["width"] for r in results_sorted_64]
        mems_64 = [r["avg_memory_mb"] for r in results_sorted_64]
        std_mems_64 = [r["std_memory_mb"] for r in results_sorted_64]

        # Step 68
        results_68 = all_scaling_results_68[model_name]
        results_sorted_68 = sorted(results_68, key=lambda x: x["num_pixels"])
        widths_68 = [r["width"] for r in results_sorted_68]
        mems_68 = [r["avg_memory_mb"] for r in results_sorted_68]
        std_mems_68 = [r["std_memory_mb"] for r in results_sorted_68]

        # Add traces for Step 64 (Left)
        fig_mem.add_trace(
            go.Scatter(
                x=widths_64,
                y=mems_64,
                error_y=dict(type="data", array=std_mems_64, visible=True),
                mode="lines+markers",
                name=f"{model_name}",
                legendgroup=model_name,
                line=dict(color=colors[model_name], width=3),
                marker=dict(size=8, symbol="circle"),
                hovertemplate="<b>%{fullData.name}</b><br>Width: %{x}<br>Memory: %{y:.1f} ± %{error_y.array:.1f}MB<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add traces for Step 68 (Right)
        fig_mem.add_trace(
            go.Scatter(
                x=widths_68,
                y=mems_68,
                error_y=dict(type="data", array=std_mems_68, visible=True),
                mode="lines+markers",
                name=f"{model_name}",
                legendgroup=model_name,
                showlegend=False,
                line=dict(color=colors[model_name], width=3),
                marker=dict(size=8, symbol="diamond"),
                hovertemplate="<b>%{fullData.name}</b><br>Width: %{x}<br>Memory: %{y:.1f} ± %{error_y.array:.1f}MB<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig_mem.update_layout(
        title="Peak Memory vs Image Width",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
    )
    fig_mem.update_xaxes(title_text="Image Width (pixels)", row=1, col=1)
    fig_mem.update_yaxes(
        title_text="Memory (MB)", range=[min_mem, max_mem], row=1, col=1
    )
    fig_mem.update_xaxes(title_text="Image Width (pixels)", row=1, col=2)
    fig_mem.update_yaxes(
        title_text="Memory (MB)", range=[min_mem, max_mem], row=1, col=2
    )

    fig_mem.write_html("image_size_vs_memory.html")
    print("Saved image_size_vs_memory.html")

    return fig_time, fig_mem


def main():
    """Main execution function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_mem_gb:.2f} GB")
    else:
        print("WARNING: Running on CPU - this will be very slow.")

    # Load models
    models = load_models(device)

    print(f"\n{'='*80}")
    print("Running Benchmarks")
    print(f"{'='*80}")

    # Scaling test for each model with varying image sizes
    all_scaling_results_64 = {}
    all_scaling_results_68 = {}

    for model_name, model in models.items():
        # Test 1: Step 64
        scaling_results_64 = benchmark_image_size(
            model, device, model_name=model_name, step=64, num_runs=3
        )
        all_scaling_results_64[model_name] = scaling_results_64

        # Test 2: Step 68
        scaling_results_68 = benchmark_image_size(
            model, device, model_name=model_name, step=68, num_runs=3
        )
        all_scaling_results_68[model_name] = scaling_results_68

    # Create Plotly visualizations
    create_scaling_plots(all_scaling_results_64, all_scaling_results_68)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
