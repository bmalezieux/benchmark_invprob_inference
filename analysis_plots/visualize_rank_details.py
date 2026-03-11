"""
Visualization script for detailed per-rank GPU metrics.

This script reads CSV files containing per-rank GPU metrics from benchmark outputs
and creates interactive Plotly visualizations showing detailed performance breakdown
for each GPU rank.

Data Format
-----------
CSV files contain 2D arrays with the structure [num_batches][n_iter]:
- gradient_time_sec: Gradient step times per batch per iteration
- denoise_time_sec: Denoising step times per batch per iteration
- gradient_memory_peak_mb: Peak GPU memory during gradient per batch per iteration
- denoise_memory_peak_mb: Peak GPU memory during denoising per batch per iteration

Times are averaged by image: sum(times across batches) / num_images
Memory is maxed across batches: max(memory across batches)

The script generates the following visualizations:
1. Gradient Time by Rank - Shows average gradient computation time for each rank
2. Denoiser Time by Rank - Shows average denoiser computation time for each rank
3. GPU Max Memory for Gradient by Rank - Shows maximum GPU memory allocated during gradient computation for each rank
4. GPU Max Memory for Denoiser by Rank - Shows maximum GPU memory allocated during denoiser computation for each rank
5. Dashboard - Combined HTML page with all 4 visualizations in a 2x2 grid

Usage
-----
From command line:
    python visualize_rank_details.py <output_dir> [results_dir]

Arguments:
    output_dir : Path to the output directory containing CSV files with GPU metrics
    results_dir : Directory to save visualization images (default: 'results_images')

Examples
--------

# Custom output directory
python analysis_plots/visualize_rank_details.py outputs/tomography_2d

Output Structure
----------------
Visualizations are saved in: results_dir/result_name/
- Individual plots: gpu_gradient_time_by_rank.html, gpu_denoiser_time_by_rank.html, etc.
- Combined dashboard: dashboard_rank_details.html
"""

import itertools
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def parse_array(val):
    """Parse array column value (string or actual array)."""
    return eval(val) if isinstance(val, str) else val

def to_2d(arr):
    """Convert 1D or scalar array to 2D format [[batch][iter]]."""
    if not isinstance(arr, (list, tuple)):
        return [[arr]]
    if len(arr) > 0 and isinstance(arr[0], (list, tuple)):
        return arr  # Already 2D
    return [arr]  # Wrap 1D in outer list

def parse_filename(filename):
    """Extract timestamp, GPU config, rank, and batch size from filename."""
    timestamp = re.search(r"(\d{8}_\d{6})", filename)
    timestamp = timestamp.group(1) if timestamp else ""
    
    bs_match = re.search(r"_bs(\d+)_", filename)
    bs = int(bs_match.group(1)) if bs_match else None
    
    if "_single_" in filename:
        return timestamp, "single", 0, 1, bs
    
    # Match patterns like "1n2t", "torchrun_2proc", etc.
    rank_match = re.search(r"_(\d+n\d+t)_rank(\d+)", filename)
    if rank_match:
        gpu_config = rank_match.group(1)
        rank = int(rank_match.group(2))
        digits = re.findall(r"\d+", gpu_config)
        num_gpus = int(digits[0]) * int(digits[1]) if len(digits) >= 2 else 1
        return timestamp, gpu_config, rank, num_gpus, bs
    
    # Match torchrun pattern: "torchrun_Nproc_rankN"
    torchrun_match = re.search(r"_torchrun_(\d+)proc_rank(\d+)", filename)
    if torchrun_match:
        num_gpus = int(torchrun_match.group(1))
        rank = int(torchrun_match.group(2))
        return timestamp, f"torchrun_{num_gpus}", rank, num_gpus, bs
    
    return timestamp, "unknown", 0, 1, bs

def read_csv_files(output_dir):
    """
    Read CSV files containing GPU metrics from the output directory.
    Each CSV row contains arrays of values for all iterations, which are expanded into separate rows.
    Groups files by configuration based on GPU count.

    Parameters
    ----------
    output_dir : str or Path
        Path to the output directory containing benchmark results

    Returns
    -------
    csv_groups : list of dict
        List of groups, where each group contains CSV files from the same run
    result_name : str
        Name of the result (from output directory)
    """
    output_path = Path(output_dir)
    csv_paths = list(output_path.glob("*gpu_metrics.csv"))

    if not csv_paths:
        print("No CSV files found")
        return [], output_path.name

    csv_data = []
    for csv_path in csv_paths:
        timestamp, gpu_config, rank, num_gpus, bs = parse_filename(csv_path.stem)

        # Read CSV and expand arrays into per-iteration rows
        raw_df = pd.read_csv(csv_path)
        expanded_rows = []
        
        for _, row in raw_df.iterrows():
            # Parse and convert all arrays to 2D format [num_batches][n_iter]
            gradient_time = to_2d(parse_array(row['gradient_time_sec']))
            denoise_time = to_2d(parse_array(row['denoise_time_sec']))
            gradient_memory_peak = to_2d(parse_array(row['gradient_memory_peak_mb']))
            denoise_memory_peak = to_2d(parse_array(row['denoise_memory_peak_mb']))
            
            num_batches = len(gradient_time)
            n_iters = len(gradient_time[0]) if num_batches > 0 else 0
            num_images = row.get('num_images', 1)
            
            # Expand into per-iteration rows with time averaged by image, memory maxed across batches
            for iter_idx in range(n_iters):
                grad_time = sum(gradient_time[b][iter_idx] for b in range(num_batches) if iter_idx < len(gradient_time[b]))
                den_time = sum(denoise_time[b][iter_idx] for b in range(num_batches) if iter_idx < len(denoise_time[b]))
                grad_mem = [gradient_memory_peak[b][iter_idx] for b in range(num_batches) if iter_idx < len(gradient_memory_peak[b])]
                den_mem = [denoise_memory_peak[b][iter_idx] for b in range(num_batches) if iter_idx < len(denoise_memory_peak[b])]
                
                expanded_rows.append({
                    'iteration': iter_idx,
                    'gradient_time_sec': grad_time / num_images if num_images > 0 else None,
                    'denoise_time_sec': den_time / num_images if num_images > 0 else None,
                    'gradient_memory_peak_mb': max(grad_mem) if grad_mem else None,
                    'denoise_memory_peak_mb': max(den_mem) if den_mem else None,
                })
        
        expanded_df = pd.DataFrame(expanded_rows)
        csv_data.append({
            "path": csv_path, "filename": csv_path.stem, "timestamp": timestamp,
            "gpu_config": gpu_config, "rank": rank, "num_gpus": num_gpus,
            "bs": bs, "df": expanded_df,
        })
        print(f"Loaded CSV file: {csv_path.name} - Expanded to {len(expanded_df)} iterations")

    # Group CSV files by (gpu_config, bs)
    get_group_key = lambda x: (x["gpu_config"], x["bs"] if x["bs"] is not None else -1)
    csv_data.sort(key=get_group_key)
    
    csv_groups = []
    for (gpu_config, _), group in itertools.groupby(csv_data, key=get_group_key):
        files = list(group)
        bs = files[0]["bs"]
        expected = 1 if gpu_config == "single" else files[0]["num_gpus"]
        
        # Take most recent file for each rank
        rank_files = {}
        for f in files:
            r = f["rank"]
            if r not in rank_files or f["timestamp"] > rank_files[r]["timestamp"]:
                rank_files[r] = f
        
        # Check if we have all ranks
        if len(rank_files) == expected and sorted(rank_files.keys()) == list(range(expected)):
            final_files = [rank_files[r] for r in sorted(rank_files.keys())]
            csv_groups.append({
                "timestamp": max(f["timestamp"] for f in final_files),
                "gpu_config": gpu_config, "bs": bs, "files": final_files,
            })
        else:
            print(f"Warning: Incomplete ranks for config={gpu_config}, bs={bs}. Found: {sorted(rank_files.keys())}, Expected: {expected}")
    
    # Sort groups by number of GPUs and batch size
    csv_groups.sort(key=lambda g: (
        1 if g["gpu_config"] == "single" else g["files"][0]["num_gpus"],
        g["bs"] if g["bs"] is not None else 0
    ))

    print(f"\nGrouped into {len(csv_groups)} configuration(s)")
    for i, group in enumerate(csv_groups):
        bs_str = f", BS={group['bs']}" if group["bs"] is not None else ""
        print(
            f"  Group {i+1}: {group['gpu_config']}{bs_str} - {len(group['files'])} file(s) - Time: {group['timestamp']}"
        )

    return csv_groups, output_path.name


def get_config_label(gpu_config):
    """Get descriptive label for GPU configuration."""
    if gpu_config == "single":
        return "1 GPU"
    if gpu_config.startswith("torchrun_"):
        num_gpus = int(gpu_config.split("_")[1])
        return f"{num_gpus} GPUs - Distributed"
    # Extract from pattern like "1n2t" -> 1*2=2
    digits = re.findall(r"\d+", gpu_config)
    if len(digits) >= 2:
        return f"{int(digits[0]) * int(digits[1])} GPUs - Distributed"
    return gpu_config

def get_group_label(group):
    """Get descriptive label for a group, including BS if present."""
    label = get_config_label(group["gpu_config"])
    return f"{label} (BS={group['bs']})" if group["bs"] is not None else label


def plot_metric_by_rank(csv_groups, output_dir, metric_config):
    """
    General function to plot any metric by rank for each configuration group.

    Parameters
    ----------
    csv_groups : list of dict
        List of CSV file groups from the same run
    output_dir : str or Path
        Directory to save the plot
    metric_config : dict
        Configuration dictionary containing:
        - 'column': str, column name in CSV
        - 'aggregation': str, 'mean' or 'max'
        - 'title': str, plot title
        - 'ylabel': str, y-axis label
        - 'hover_label': str, label for hover template
        - 'format': str, format string for hover value (e.g., '.3f' or '.2f')
        - 'filename': str, output filename
    """
    fig = go.Figure()

    # Determine layout for manual positioning
    if not csv_groups:
        return

    # Calculate global layout parameters to ensure consistent bar widths
    max_ranks = max(len(group["files"]) for group in csv_groups)
    bar_width = 0.2
    # Ensure group spacing is large enough to hold the widest group plus padding
    group_spacing = max(1.0, max_ranks * bar_width * 1.25)

    x_tick_vals = []
    x_tick_text = []

    for group_idx, group in enumerate(csv_groups):
        config_label = get_group_label(group)
        files = sorted(group["files"], key=lambda x: x["rank"])
        num_ranks = len(files)

        # Center x-position for this group
        group_center_x = group_idx * group_spacing
        x_tick_vals.append(group_center_x)
        x_tick_text.append(config_label)

        for i, file_info in enumerate(files):
            df = file_info["df"]
            rank = file_info["rank"]

            # Aggregate metric value
            if metric_config["aggregation"] == "mean":
                metric_value = df[metric_config["column"]].mean()
            elif metric_config["aggregation"] == "max":
                metric_value = df[metric_config["column"]].max()
            else:
                raise ValueError(
                    f"Unsupported aggregation: {metric_config['aggregation']}"
                )

            if len(group["files"]) > 1:
                label = f"{config_label} - Rank {rank}"
            else:
                label = config_label

            # Calculate manual bar position to center the group
            # Formula: center + (index - (count-1)/2) * width
            bar_x = group_center_x + (i - (num_ranks - 1) / 2) * bar_width

            fig.add_trace(
                go.Bar(
                    name=label,
                    x=[bar_x],
                    y=[metric_value],
                    width=[bar_width],
                    text=[f"Rank {rank}" if len(group["files"]) > 1 else ""],
                    textposition="auto",
                    hovertemplate=f"<b>{label}</b><br>"
                    + f"<b>{metric_config['hover_label']}:</b> %{{y:{metric_config['format']}}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(text=metric_config["title"], font=dict(size=20)),
        xaxis=dict(
            title="Configuration",
            tickmode="array",
            tickvals=x_tick_vals,
            ticktext=x_tick_text,
            tickfont=dict(size=14),
        ),
        yaxis_title=metric_config["ylabel"],
        template="plotly_white",
        margin=dict(l=50, r=20, t=60, b=80),
        showlegend=False,
        font=dict(size=14),
    )

    output_path = Path(output_dir) / metric_config["filename"]
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def create_dashboard(results_path, result_name):
    """Create an HTML dashboard combining all visualizations in a 2x2 grid."""
    required_files = [
        "gpu_gradient_time_by_rank.html",
        "gpu_denoiser_time_by_rank.html",
        "gpu_gradient_memory_by_rank.html",
        "gpu_denoiser_memory_by_rank.html",
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
    <title>Per-Rank GPU Metrics Dashboard - {result_name}</title>
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
            <h1>📊 Per-Rank GPU Metrics Dashboard</h1>
            <p>{result_name.replace('_', ' ').title()}</p>
        </div>
        <div class="timestamp">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</div>
        <div class="dashboard-grid">
            <div class="chart-container"><iframe src="gpu_gradient_time_by_rank.html" title="Gradient Time"></iframe></div>
            <div class="chart-container"><iframe src="gpu_denoiser_time_by_rank.html" title="Denoiser Time"></iframe></div>
            <div class="chart-container"><iframe src="gpu_gradient_memory_by_rank.html" title="Gradient Memory"></iframe></div>
            <div class="chart-container"><iframe src="gpu_denoiser_memory_by_rank.html" title="Denoiser Memory"></iframe></div>
        </div>
        <div class="footer">Per-Rank GPU Metrics Dashboard | Generated by visualize_rank_details.py</div>
    </div>
</body>
</html>"""

    dashboard_path = results_path / "dashboard_rank_details.html"
    with open(dashboard_path, "w") as f:
        f.write(dashboard_html)

    print(f"Dashboard created: {dashboard_path}")
    return dashboard_path


def visualize_rank_details(output_dir, results_dir="results_images"):
    """
    Main function to create all per-rank detailed visualizations.

    Parameters
    ----------
    output_dir : str or Path
        Path to the output directory containing CSV files
    results_dir : str or Path
        Directory to save visualization images (default: 'results_images')
    """
    print(f"Reading CSV files from: {output_dir}")
    print("-" * 60)

    # Read CSV data
    csv_groups, result_name = read_csv_files(output_dir)

    if not csv_groups:
        print("No CSV files found or no complete groups, skipping visualization")
        return

    # Create results directory with result name subdirectory
    results_path = Path(results_dir) / result_name
    results_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving visualizations to: {results_path}")
    print("-" * 60)
    print("Creating per-rank visualizations...")
    print("-" * 60)

    # Create plots from CSV data
    plot_metric_by_rank(
        csv_groups,
        results_path,
        {
            "column": "gradient_time_sec",
            "aggregation": "mean",
            "title": "Average Gradient Time by Rank",
            "ylabel": "Average Gradient Time (seconds)",
            "hover_label": "Avg Gradient Time",
            "format": ".3f",
            "filename": "gpu_gradient_time_by_rank.html",
        },
    )

    plot_metric_by_rank(
        csv_groups,
        results_path,
        {
            "column": "denoise_time_sec",
            "aggregation": "mean",
            "title": "Average Denoiser Time by Rank",
            "ylabel": "Average Denoiser Time (seconds)",
            "hover_label": "Avg Denoiser Time",
            "format": ".3f",
            "filename": "gpu_denoiser_time_by_rank.html",
        },
    )

    plot_metric_by_rank(
        csv_groups,
        results_path,
        {
            "column": "gradient_memory_peak_mb",
            "aggregation": "max",
            "title": "GPU Max Memory for Gradient by Rank",
            "ylabel": "Max Memory Allocated (MB)",
            "hover_label": "Max Gradient Memory",
            "format": ".2f",
            "filename": "gpu_gradient_memory_by_rank.html",
        },
    )

    plot_metric_by_rank(
        csv_groups,
        results_path,
        {
            "column": "denoise_memory_peak_mb",
            "aggregation": "max",
            "title": "GPU Max Memory for Denoiser by Rank",
            "ylabel": "Max Memory Allocated (MB)",
            "hover_label": "Max Denoiser Memory",
            "format": ".2f",
            "filename": "gpu_denoiser_memory_by_rank.html",
        },
    )

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
        description="Visualize detailed per-rank GPU metrics from CSV files"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="outputs/highres_color_image",
        help="Path to the output directory containing CSV files",
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="docs/source/_static/images/",
        help="Directory to save visualization images",
    )

    args = parser.parse_args()

    visualize_rank_details(args.output_dir, args.results_dir)
