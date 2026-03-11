"""
Visualization script for general benchmark performance results.

This script reads parquet files from benchmark outputs and creates
interactive Plotly visualizations for overall performance metrics.

The script generates the following visualizations:
1. PSNR vs Time - Shows how PSNR evolves over computation time
2. PSNR vs Iteration - Shows PSNR convergence over iterations
3. Time Breakdown - Stacked bar chart showing average time spent in gradient and denoiser operations
4. GPU Memory Usage - Bar chart showing average GPU memory for gradient and denoiser
5. Dashboard - Combined HTML page with all 4 visualizations in a 2x2 grid

Usage
-----
From command line:
    python analysis_plots/visualize_general_results.py <output_dir> [results_dir]

Arguments:
    output_dir : Path to the output directory containing benchmark results (parquet file)
    results_dir : Directory to save visualization images (default: 'results_images')

Examples
--------
python analysis_plots/visualize_general_results.py outputs/tomography_2d

Output Structure
----------------
Visualizations are saved in: results_dir/result_name/
- Individual plots: psnr_vs_time.html, psnr_vs_iteration.html, time_breakdown_stacked.html, gpu_memory_gradient_denoiser.html
- Combined dashboard: dashboard.html

"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_config_label(row):
    """Create a descriptive label for a configuration based on key parameters."""
    gres = row.get("p_solver_slurm_gres", "gpu:1")
    ngpu = int(gres.split(":")[1]) if isinstance(gres, str) and ":" in gres else 1

    max_batch = row.get("p_solver_max_batch_size", None)

    if max_batch is not None and max_batch > 0:
        label = f"{ngpu} GPU{'s' if ngpu > 1 else ''} - Batch={max_batch}"
    else:
        label = f"{ngpu} GPU{'s' if ngpu > 1 else ''}"
        dist_phy = row.get("p_solver_distribute_physics", False)
        dist_den = row.get("p_solver_distribute_denoiser", False)

        if dist_phy and dist_den:
            label += " - Distributed"
        elif dist_phy:
            label += " - Dist Physics"
        elif dist_den:
            label += " - Dist Denoiser"

    return label


def read_parquet_data(output_dir):
    """Read parquet file and expand 2D arrays into per-iteration rows."""
    output_path = Path(output_dir)
    parquet_files = list(output_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file found in {output_dir}")

    parquet_df = pd.read_parquet(parquet_files[0])
    
    def get_ngpu(row):
        gres = row.get("p_solver_slurm_gres", "gpu:1")
        return int(gres.split(":")[1]) if isinstance(gres, str) and ":" in gres else 1

    parquet_df["ngpu"] = parquet_df.apply(get_ngpu, axis=1)
    parquet_df["batch"] = parquet_df["p_solver_max_batch_size"].fillna(0)
    parquet_df["config_label"] = parquet_df.apply(create_config_label, axis=1)
    
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
    
    def max_across_batches(data, batch_idx, iter_idx):
        """Max metric across batches for given iteration."""
        if not data or not isinstance(data, list):
            return None
        values = [data[b][iter_idx] for b in range(len(data)) if len(data[b]) > iter_idx]
        return max(values) if values else None
    
    expanded_rows = []
    
    for idx, row in parquet_df.iterrows():
        psnr_iter = to_list(row.get('objective_psnr_per_image_iter', None))
        init_psnr = to_list(row.get('objective_init_psnr_per_image', None))
        iter_times = to_list(row.get('objective_iter_times_per_batch', None))
        init_times = to_list(row.get('objective_init_time_per_batch', None))
        grad_times = to_list(row.get('objective_gradient_time_sec', None))
        denoise_times = to_list(row.get('objective_denoise_time_sec', None))
        grad_mem = to_list(row.get('objective_gradient_memory_peak_mb', None))
        denoise_mem = to_list(row.get('objective_denoise_memory_peak_mb', None))
        
        if not psnr_iter or not isinstance(psnr_iter, list) or len(psnr_iter) == 0:
            continue
        
        num_images = len(psnr_iter)
        n_iters = len(psnr_iter[0]) if num_images > 0 else 0
        
        if n_iters == 0:
            continue
        
        # Iteration 0 (init)
        if init_psnr and isinstance(init_psnr, list) and len(init_psnr) > 0:
            new_row = row.copy()
            new_row['iteration'] = 0
            new_row['objective_psnr'] = np.mean(init_psnr)
            new_row['objective_psnr_std'] = np.std(init_psnr)
            new_row['objective_iter_time'] = np.sum(init_times) / num_images if init_times else None
            new_row['objective_gradient_time'] = None
            new_row['objective_denoise_time'] = None
            new_row['objective_gradient_memory_peak'] = None
            new_row['objective_denoise_memory_peak'] = None
            expanded_rows.append(new_row)
        
        # Iterations 1..n
        for iter_idx in range(n_iters):
            new_row = row.copy()
            new_row['iteration'] = iter_idx + 1
            
            psnr_vals = [psnr_iter[i][iter_idx] for i in range(num_images) if len(psnr_iter[i]) > iter_idx]
            new_row['objective_psnr'] = np.mean(psnr_vals) if psnr_vals else None
            new_row['objective_psnr_std'] = np.std(psnr_vals) if psnr_vals else 0.0
            new_row['objective_iter_time'] = avg_time_per_image(iter_times, num_images, iter_idx)
            new_row['objective_gradient_time'] = avg_time_per_image(grad_times, num_images, iter_idx)
            new_row['objective_denoise_time'] = avg_time_per_image(denoise_times, num_images, iter_idx)
            new_row['objective_gradient_memory_peak'] = max_across_batches(grad_mem, 0, iter_idx)
            new_row['objective_denoise_memory_peak'] = max_across_batches(denoise_mem, 0, iter_idx)
            expanded_rows.append(new_row)
    
    if not expanded_rows:
        print("Warning: No per-iteration data found, returning original dataframe")
        return parquet_df, output_path.name
    
    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df.sort_values(by=["ngpu", "batch", "iteration"], inplace=True)

    result_name = output_path.name

    print(f"Loaded parquet file: {parquet_files[0].name}")
    print(f"Original shape: {parquet_df.shape}, Expanded shape: {expanded_df.shape}")
    print("\nConfigurations found:")
    for label in expanded_df["config_label"].unique():
        config_data = expanded_df[expanded_df["config_label"] == label]
        n_iters = config_data["iteration"].max()
        n_runs = len(config_data) // n_iters if n_iters > 0 else 0
        print(f"  - {label}: {n_runs} run(s) x {n_iters} iterations")

    return expanded_df, result_name


def create_plot_layout(
    title, xaxis_title, yaxis_title, show_legend=False, legend_y=-0.2
):
    """Create standard layout for plots."""
    layout = dict(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="closest",
        template="plotly_white",
        font=dict(size=14),
        showlegend=show_legend,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    if show_legend:
        layout["legend"] = dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        )
    return layout


def plot_psnr_vs_metric(
    parquet_df, output_dir, metric="time", filename="psnr_vs_time.html", show_std=False
):
    """Plot PSNR vs time or iteration with optional std deviation."""
    fig = go.Figure()
    groups = parquet_df.groupby("config_label")
    sorted_labels = (
        parquet_df[["config_label", "ngpu", "batch"]]
        .drop_duplicates()
        .sort_values(by=["ngpu", "batch"])["config_label"]
    )

    metric_col = "time" if metric == "time" else "iteration"
    metric_label = "Time" if metric == "time" else "Iteration"
    metric_format = ":.3f" if metric == "time" else ""

    for label in sorted_labels:
        group = groups.get_group(label).sort_values(metric_col)
        
        # Compute cumulative time if metric is time
        if metric == "time":
            group = group.copy()
            group['time'] = group['objective_iter_time'].cumsum()
        
        if show_std and 'objective_psnr_std' in group.columns:
            fig.add_trace(
                go.Scatter(
                    x=group[metric_col],
                    y=group["objective_psnr"],
                    mode="lines+markers",
                    name=label,
                    line=dict(width=4),
                    marker=dict(size=10),
                    error_y=dict(type='data', array=group['objective_psnr_std'], visible=True),
                    hovertemplate=f"<b>Config:</b> {label}<br><b>{metric_label}:</b> %{{x{metric_format}}}<br><b>PSNR:</b> %{{y:.2f}} ± %{{error_y.array:.2f}} dB<br><extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=group[metric_col],
                    y=group["objective_psnr"],
                    mode="lines+markers",
                    name=label,
                    line=dict(width=4),
                    marker=dict(size=10),
                    hovertemplate=f"<b>Config:</b> {label}<br><b>{metric_label}:</b> %{{x{metric_format}}}<br><b>PSNR:</b> %{{y:.2f}} dB<br><extra></extra>",
                )
            )

    title = f"PSNR vs {metric_label}"
    xaxis = f"{metric_label}" + (" (seconds)" if metric == "time" else "")
    show_legend = metric == "time"
    fig.update_layout(
        **create_plot_layout(title, xaxis, "PSNR (dB)", show_legend=show_legend, legend_y=-0.15)
    )

    output_path = Path(output_dir) / filename
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")
    return fig


def plot_time_breakdown_stacked(parquet_df, output_dir):
    """Plot stacked bar chart showing average time for gradient and denoiser."""
    agg_data = parquet_df[parquet_df['iteration'] > 0].groupby("config_label").agg({
        "objective_gradient_time": "mean",
        "objective_denoise_time": "mean",
        "ngpu": "first",
        "batch": "first",
    }).reset_index()

    agg_data["total_time"] = agg_data["objective_gradient_time"] + agg_data["objective_denoise_time"]
    agg_data["denoiser_pct"] = (agg_data["objective_denoise_time"] / agg_data["total_time"]) * 100
    agg_data.sort_values(by=["ngpu", "batch"], inplace=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Gradient",
            x=agg_data["config_label"],
            y=agg_data["objective_gradient_time"],
            marker_color="blue",
            width=0.5,
            hovertemplate="<b>Gradient Time:</b> %{y:.3f}s<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Denoiser",
            x=agg_data["config_label"],
            y=agg_data["objective_denoise_time"],
            marker_color="orange",
            text=agg_data["denoiser_pct"],
            texttemplate="%{text:.1f}%",
            textposition="auto",
            textfont=dict(size=12),
            width=0.5,
            hovertemplate="<b>Denoiser Time:</b> %{y:.3f}s (%{text:.1f}%)<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=agg_data["config_label"],
            y=agg_data["total_time"],
            mode="text",
            text=agg_data["total_time"],
            texttemplate="%{text:.1f}s",
            textposition="top center",
            textfont=dict(size=12),
            cliponaxis=False,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=dict(text="Average Time Breakdown per Step", font=dict(size=20)),
        xaxis_title="Configuration",
        yaxis=dict(
            title="Average Time (seconds)",
            range=[0, agg_data["total_time"].max() * 1.2],
        ),
        xaxis=dict(tickfont=dict(size=14)),
        barmode="stack",
        template="plotly_white",
        margin=dict(t=60, b=60, l=50, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        hovermode="x unified",
        font=dict(size=14),
    )

    output_path = Path(output_dir) / "time_breakdown_stacked.html"
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")
    return fig


def plot_gpu_memory_from_parquet(parquet_df, output_dir):
    """Plot GPU max memory allocated with gradient/denoiser breakdown."""
    agg_data = parquet_df[parquet_df['iteration'] > 0].groupby("config_label").agg({
        "objective_gradient_memory_peak": "mean",
        "objective_denoise_memory_peak": "mean",
        "ngpu": "first",
        "batch": "first",
    }).reset_index()

    agg_data.sort_values(by=["ngpu", "batch"], inplace=True)

    fig = go.Figure()

    for name, col, color in [
        ("Gradient", "objective_gradient_memory_peak", "blue"),
        ("Denoiser", "objective_denoise_memory_peak", "orange"),
    ]:
        fig.add_trace(
            go.Bar(
                name=name,
                x=agg_data["config_label"],
                y=agg_data[col],
                marker_color=color,
                text=agg_data[col],
                texttemplate="%{y:.1f} MB",
                textposition="auto",
                textfont=dict(size=12),
                hovertemplate=f"<b>{name} Memory:</b> %{{y:.2f}} MB<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="Average GPU Max Memory Allocated", font=dict(size=20)),
        xaxis_title="Configuration",
        yaxis_title="Max Memory Allocated (MB)",
        xaxis=dict(tickfont=dict(size=14)),
        barmode="group",
        template="plotly_white",
        margin=dict(t=60, b=60, l=50, r=20),
        showlegend=False,
        font=dict(size=14),
    )

    output_path = Path(output_dir) / "gpu_memory_gradient_denoiser.html"
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")
    return fig


def create_dashboard(results_path, result_name):
    """Create an HTML dashboard combining all visualizations in a 2x2 grid."""
    required_files = [
        "psnr_vs_time.html",
        "psnr_vs_iteration.html",
        "time_breakdown_stacked.html",
        "gpu_memory_gradient_denoiser.html",
    ]

    for file in required_files:
        if not (results_path / file).exists():
            print(f"Warning: Required file not found: {results_path / file}")
            return None

    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Results Dashboard - {result_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{ 
            width: 100%; 
            height: 100%; 
            margin: 0; 
            padding: 0; 
            overflow-x: hidden;
        }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: white;
        }}
        .container {{ 
            width: 100%; 
            min-height: 100%;
            background: white;
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 20px; 
            text-align: center;
        }}
        .header h1 {{ font-size: clamp(1.5em, 4vw, 2.5em); margin-bottom: 8px; font-weight: 700; }}
        .header p {{ font-size: clamp(0.9em, 2vw, 1.2em); opacity: 0.95; }}
        .timestamp {{ 
            text-align: center; 
            padding: 10px; 
            background: #f8f9fa;
            color: #6c757d; 
            font-size: clamp(0.8em, 1.5vw, 0.9em);
        }}
        .dashboard-grid {{ 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 0;
            width: 100%;
        }}
        .chart-container {{ 
            width: 100%; 
            height: 0;
            padding-bottom: 90%; /* Aspect ratio control */
            position: relative;
            background: white; 
            border: 1px solid #e0e0e0;
        }}
        iframe {{ 
            position: absolute;
            top: 0;
            left: 0;
            width: 100%; 
            height: 100%; 
            border: none; 
            display: block; 
        }}
        .footer {{ 
            text-align: center; 
            padding: 15px; 
            background: #f8f9fa; 
            color: #6c757d; 
            font-size: clamp(0.8em, 1.5vw, 0.9em); 
        }}
        
        @media (max-width: 768px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
            .chart-container {{
                padding-bottom: 80%;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Benchmark Results Dashboard</h1>
            <p>{result_name.replace('_', ' ').title()}</p>
        </div>
        <div class="timestamp">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</div>
        <div class="dashboard-grid">
            <div class="chart-container"><iframe src="psnr_vs_time.html" title="PSNR vs Time"></iframe></div>
            <div class="chart-container"><iframe src="psnr_vs_iteration.html" title="PSNR vs Iteration"></iframe></div>
            <div class="chart-container"><iframe src="time_breakdown_stacked.html" title="Time Breakdown"></iframe></div>
            <div class="chart-container"><iframe src="gpu_memory_gradient_denoiser.html" title="GPU Memory"></iframe></div>
        </div>
        <div class="footer">Benchmark Analysis Dashboard </div>
    </div>
</body>
</html>"""

    dashboard_path = results_path / "dashboard.html"
    with open(dashboard_path, "w") as f:
        f.write(dashboard_html)

    print(f"Dashboard created: {dashboard_path}")
    return dashboard_path


def visualize_general_results(output_dir, results_dir="results_images"):
    """Main function to create all general performance visualizations."""
    print(f"Reading data from: {output_dir}")
    print("-" * 60)

    parquet_df, result_name = read_parquet_data(output_dir)

    results_path = Path(results_dir) / result_name
    results_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving visualizations to: {results_path}")
    print("-" * 60)
    print("Creating visualizations...")
    print("-" * 60)

    # Create all plots
    plot_psnr_vs_metric(
        parquet_df, results_path, metric="time", filename="psnr_vs_time.html"
    )
    plot_psnr_vs_metric(
        parquet_df, results_path, metric="iteration", filename="psnr_vs_iteration.html"
    )
    plot_time_breakdown_stacked(parquet_df, results_path)
    plot_gpu_memory_from_parquet(parquet_df, results_path)

    print("-" * 60)
    print("Creating dashboard...")
    print("-" * 60)

    dashboard_path = create_dashboard(results_path, result_name)

    print("-" * 60)
    print("All visualizations completed!")
    if dashboard_path:
        print(f"✨ View your dashboard at: {dashboard_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize general benchmark results from parquet file"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="outputs/highres_color_image",
        help="Path to the output directory containing benchmark results",
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="docs/source/_static/images/",
        help="Directory to save visualization images",
    )

    args = parser.parse_args()

    visualize_general_results(args.output_dir, args.results_dir)
