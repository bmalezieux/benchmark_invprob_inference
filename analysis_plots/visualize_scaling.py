"""
Visualization script for scaling benchmark results.

This script reads parquet files from benchmark scaling outputs and creates
interactive Plotly visualizations for strong scaling and parallel efficiency.

The script generates the following visualizations:
1. Strong Scaling Speedup - Shows speedup vs number of GPUs (with legend)
2. Strong Scaling Speedup Gradient - Shows gradient speedup vs number of GPUs (no legend)
3. Strong Scaling Speedup Denoiser - Shows denoiser speedup vs number of GPUs (no legend)
4. Parallel Efficiency - Shows efficiency vs number of GPUs (with legend)
5. Dashboard - Combined HTML page with all 4 visualizations in a 2x2 grid

Usage
-----
From command line:
    python visualize_scaling.py <output_dir> [results_dir]

Arguments:
    output_dir : Path to the output directory containing benchmark results (parquet file)
    results_dir : Directory to save visualization images (default: 'results_images')

Examples
--------

# Default output directory
python analysis_plots/visualize_scaling.py outputs/scaling/highres_color_image

Output Structure
----------------
Visualizations are saved in: results_dir/scaling_result_name/
- Individual plots: strong_scaling_speedup.html, parallel_efficiency.html, etc.
- Combined dashboard: dashboard_scaling.html

"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def to_list(arr):
    """Convert numpy array to list recursively."""
    if isinstance(arr, np.ndarray):
        return [to_list(x) for x in arr] if arr.ndim > 1 else arr.tolist()
    return arr

def avg_time_per_image(data, num_images, iter_idx):
    """Average time per image by summing across batches and dividing by number of images."""
    if not data or not isinstance(data, list) or num_images == 0:
        return None
    values = [data[b][iter_idx] for b in range(len(data)) if len(data[b]) > iter_idx]
    return np.sum(values) / num_images if values else None

def read_parquet_data(output_dir):
    output_path = Path(output_dir)
    parquet_files = list(output_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file found in {output_dir}")

    parquet_df = pd.read_parquet(parquet_files[0])

    # Helper functions
    get_total_gpus = lambda row: (
        row.get("p_solver_slurm_nodes", 1) * 
        (int(row.get("p_solver_slurm_gres", "gpu:1").split(":")[1]) 
         if isinstance(row.get("p_solver_slurm_gres", "gpu:1"), str) and ":" in row.get("p_solver_slurm_gres", "gpu:1") else 1)
    )
    
    def create_config_label(row):
        nodes, tasks = row.get("p_solver_slurm_nodes", 1), row.get("p_solver_slurm_ntasks_per_node", 1)
        return "Single GPU" if nodes == 1 and tasks == 1 else (f"{nodes}n{tasks}t" if nodes > 1 else f"1n{tasks}t")

    parquet_df["ngpu"] = parquet_df.apply(get_total_gpus, axis=1)
    parquet_df["config_label"] = parquet_df.apply(create_config_label, axis=1)
    
    # Expand per-iteration 2D arrays into separate rows
    expanded_rows = []
    
    for idx, row in parquet_df.iterrows():
        # New format: 2D arrays
        psnr_iter = to_list(row.get('objective_psnr_per_image_iter', None))
        iter_times = to_list(row.get('objective_iter_times_per_batch', None))
        grad_times = to_list(row.get('objective_gradient_time_sec', None))
        denoise_times = to_list(row.get('objective_denoise_time_sec', None))
        
        if not psnr_iter or not isinstance(psnr_iter, list) or len(psnr_iter) == 0:
            continue
        
        num_images = len(psnr_iter)
        n_iters = len(psnr_iter[0]) if num_images > 0 else 0
        
        if n_iters == 0:
            continue
        
        # Expand iterations
        for iter_idx in range(n_iters):
            psnr_vals = [psnr_iter[i][iter_idx] for i in range(num_images) if len(psnr_iter[i]) > iter_idx]
            new_row = row.copy()
            new_row['iteration'] = iter_idx + 1
            new_row['objective_psnr'] = np.mean(psnr_vals) if psnr_vals else None
            new_row['objective_iter_time'] = avg_time_per_image(iter_times, num_images, iter_idx)
            new_row['objective_gradient_time'] = avg_time_per_image(grad_times, num_images, iter_idx)
            new_row['objective_denoise_time'] = avg_time_per_image(denoise_times, num_images, iter_idx)
            expanded_rows.append(new_row)
    
    if not expanded_rows:
        print("Warning: No per-iteration data found, returning original dataframe")
        return parquet_df, output_path.name
    
    expanded_df = pd.DataFrame(expanded_rows)
    # Compute cumulative time across iterations for each configuration
    expanded_df['cumulative_time'] = expanded_df.groupby(['config_label', 'idx_rep'])['objective_iter_time'].cumsum()
    
    # Detect image size column
    img_size_col = next(
        (col for col in ["p_dataset_image_size", "p_dataset_img_size", "image_size"] 
         if col in expanded_df.columns), None
    )
    if not img_size_col:
        img_size_col = "image_size"
        expanded_df[img_size_col] = "Default"
    expanded_df.attrs["img_size_col"] = img_size_col

    print(f"Loaded: {parquet_files[0].name}, Original: {parquet_df.shape}, Expanded: {expanded_df.shape}")
    return expanded_df, output_path.name


def calculate_scaling_metrics(parquet_df):
    img_size_col = parquet_df.attrs.get("img_size_col")
    scaling_dict = {}

    # Find common max iteration across all configs
    common_max_iter = parquet_df.groupby("ngpu")["iteration"].max().min()
    print(f"Using iteration {common_max_iter} for analysis")

    filtered_df = parquet_df[parquet_df["iteration"] <= common_max_iter]

    for img_size in sorted(filtered_df[img_size_col].unique()):
        img_df = filtered_df[filtered_df[img_size_col] == img_size]
        
        scaling_data = [
            {
                "ngpu": ngpu,
                "config_label": max_iter_data["config_label"].values[0],
                "total_time": max_iter_data["cumulative_time"].values[0],
                "gradient_time": gpu_data["objective_gradient_time"].mean(),
                "denoiser_time": gpu_data["objective_denoise_time"].mean(),
            }
            for ngpu in sorted(img_df["ngpu"].unique())
            if not (max_iter_data := (gpu_data := img_df[img_df["ngpu"] == ngpu])[
                gpu_data["iteration"] == common_max_iter
            ]).empty
        ]

        if scaling_data:
            sdf = pd.DataFrame(scaling_data)
            base = sdf.loc[sdf["ngpu"].idxmin()]

            sdf["speedup"] = base["total_time"] / sdf["total_time"]
            sdf["gradient_speedup"] = base["gradient_time"] / sdf["gradient_time"]
            sdf["denoiser_speedup"] = base["denoiser_time"] / sdf["denoiser_time"]
            sdf["ideal_speedup"] = sdf["ngpu"] / base["ngpu"]
            sdf["efficiency"] = (sdf["speedup"] / sdf["ideal_speedup"]) * 100
            sdf.loc[sdf["ngpu"] == base["ngpu"], "efficiency"] = 100.0

            scaling_dict[img_size] = sdf

    return scaling_dict


def plot_generic(scaling_dict, output_path, title, y_label, metric_cfg, y_range=None):
    """
    Generic plotting function.
    metric_cfg:
      - str: column name (plots one line per image size)
      - dict: {col: (color, label)} (plots multiple lines for single image size)
    """
    fig = go.Figure()
    all_ngpus = sorted(list({n for df in scaling_dict.values() for n in df["ngpu"]}))

    # Add traces
    if isinstance(metric_cfg, str):
        colors = ["blue", "green", "red", "purple", "orange", "brown"]
        for idx, (img_size, df) in enumerate(sorted(scaling_dict.items())):
            label = (
                f"Image {img_size}×{img_size}"
                if isinstance(img_size, (int, float))
                else str(img_size)
            )
            fig.add_trace(
                go.Scatter(
                    x=df["ngpu"],
                    y=df[metric_cfg],
                    mode="lines+markers",
                    name=label,
                    line=dict(width=4, color=colors[idx % len(colors)]),
                    marker=dict(size=12),
                    hovertemplate=f"<b>Image:</b> {img_size}<br><b>GPUs:</b> %{{x}}<br><b>{y_label}:</b> %{{y:.2f}}<extra></extra>",
                )
            )

    elif isinstance(metric_cfg, dict):
        # Assumes single image size dict
        df = list(scaling_dict.values())[0]
        for col, (color, label) in metric_cfg.items():
            fig.add_trace(
                go.Scatter(
                    x=df["ngpu"],
                    y=df[col],
                    mode="lines+markers",
                    name=label,
                    line=dict(width=3, color=color),
                    marker=dict(size=8),
                    hovertemplate=f"<b>GPUs:</b> %{{x}}<br><b>{label}:</b> %{{y:.2f}}<extra></extra>",
                )
            )

    # Add Ideal Line
    if all_ngpus:
        if "efficiency" in y_label.lower():
            ideal = [100] * len(all_ngpus)
            name = "Ideal Efficiency"
        else:
            baseline = min(all_ngpus)
            ideal = [n / baseline for n in all_ngpus]
            name = "Ideal Speedup"

        fig.add_trace(
            go.Scatter(
                x=all_ngpus,
                y=ideal,
                mode="lines",
                name=name,
                line=dict(width=3, color="black", dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis=dict(
            title="Number of GPUs",
            tickmode="array",
            tickvals=all_ngpus,
            tickfont=dict(size=14),
        ),
        yaxis=dict(title=y_label, range=y_range, tickfont=dict(size=14)),
        template="plotly_white",
        margin=dict(l=50, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
        hovermode="x unified",
    )

    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def create_dashboard(results_path, result_name, plots):
    charts_html = "".join(
        [
            f'<div class="chart-container"><iframe src="{p}" title="{p}"></iframe></div>'
            for p in plots
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Scaling Results - {result_name}</title>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .dashboard-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
        .chart-container {{ position: relative; padding-bottom: 70%; background: white; border: 0; }}
        iframe {{ position: absolute; width: 100%; height: 100%; border: none; display: block; }}
    </style>
</head>
<body>
    <div class="header"><h1>📊 Scaling Results: {result_name}</h1><p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p></div>
    <div class="dashboard-grid">{charts_html}</div>
</body>
</html>"""

    path = results_path / "dashboard_scaling.html"
    with open(path, "w") as f:
        f.write(html)
    print(f"Dashboard: {path}")


def visualize_scaling(output_dir, results_dir="results_images"):
    print(f"Processing: {output_dir}")
    try:
        df, name = read_parquet_data(output_dir)
        scaling_dict = calculate_scaling_metrics(df)
        if not scaling_dict:
            return

        out_path = Path(results_dir) / f"scaling_{name}"
        out_path.mkdir(exist_ok=True, parents=True)

        plots_created = []

        # Parallel Efficiency (Always created)
        plot_generic(
            scaling_dict,
            out_path / "parallel_efficiency.html",
            "Parallel Efficiency",
            "Efficiency (%)",
            "efficiency",
            y_range=[0, 110],
        )
        plots_created.append("parallel_efficiency.html")

        if len(scaling_dict) == 1:
            # Single Image Case: Combined Plot
            cfg = {
                "speedup": ("green", "Total"),
                "gradient_speedup": ("blue", "Gradient"),
                "denoiser_speedup": ("orange", "Denoiser"),
            }
            plot_generic(
                scaling_dict,
                out_path / "combined_strong_scaling.html",
                "Strong Scaling Breakdown",
                "Speedup",
                cfg,
            )
            plots_created.insert(0, "combined_strong_scaling.html")  # Put first
        else:
            # Multi Image Case: Separate Plots
            for metric, title in [
                ("speedup", "Strong Scaling"),
                ("gradient_speedup", "Gradient Scaling"),
                ("denoiser_speedup", "Denoiser Scaling"),
            ]:
                fname = (
                    f"strong_scaling_{metric.split('_')[0]}_speedup.html"
                    if "_" in metric
                    else "strong_scaling_speedup.html"
                )
                plot_generic(scaling_dict, out_path / fname, title, "Speedup", metric)
                plots_created.append(fname)

            # Reorder for dashboard: Total, Efficiency, Gradient, Denoiser
            plots_created = [
                "strong_scaling_speedup.html",
                "parallel_efficiency.html",
                "strong_scaling_gradient_speedup.html",
                "strong_scaling_denoiser_speedup.html",
            ]

        create_dashboard(out_path, name, plots_created)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", nargs="?", default="outputs/scaling/highres_color_image"
    )
    parser.add_argument("results_dir", nargs="?", default="docs/source/_static/images/")
    args = parser.parse_args()
    visualize_scaling(args.output_dir, args.results_dir)
