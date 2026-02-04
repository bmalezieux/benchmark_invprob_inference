"""
Visualization script for detailed per-rank GPU metrics.

This script reads CSV files containing per-rank GPU metrics from benchmark outputs 
and creates interactive Plotly visualizations showing detailed performance breakdown 
for each GPU rank.

The script generates the following visualizations:
1. Gradient Time by Rank - Shows average gradient computation time for each rank
2. Denoiser Time by Rank - Shows average denoiser computation time for each rank
3. GPU Max Memory for Gradient by Rank - Shows maximum GPU memory allocated during gradient computation for each rank
4. GPU Max Memory for Denoiser by Rank - Shows maximum GPU memory allocated during denoiser computation for each rank

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
where result_name is the name of the output directory (e.g., 'highres_color_image')
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import re
import itertools
from datetime import datetime, timedelta


def read_csv_files(output_dir):
    """
    Read CSV files containing GPU metrics from the output directory.
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
    
    # Find and read all CSV files
    csv_paths = list(output_path.glob("*gpu_metrics.csv"))
    
    if not csv_paths:
        print("No CSV files found")
        return [], output_path.name
    
    # Parse filenames to extract timestamp and GPU configuration
    csv_data = []
    for csv_path in csv_paths:
        filename = csv_path.stem
        
        # Extract timestamp 
        timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
        timestamp = timestamp_match.group(1) if timestamp_match else ""
        
        # Determine GPU configuration
        if '_single_' in filename:
            gpu_config = 'single'
            rank = 0
            num_gpus = 1
        else:
            # Extract pattern before '_rank' (e.g., "1n2t", "1n4t", "2n2t")
            rank_match = re.search(r'_(\d+n\d+t)_rank(\d+)', filename)
            if rank_match:
                gpu_config = rank_match.group(1)
                rank = int(rank_match.group(2))
                # Extract number of GPUs: first digit * third digit (e.g., "1n2t" -> 1*2=2, "1n4t" -> 1*4=4)
                digits = re.findall(r'\d+', gpu_config)
                if len(digits) >= 2:
                    num_gpus = int(digits[0]) * int(digits[1])
                else:
                    num_gpus = 1
            else:
                gpu_config = 'unknown'
                rank = 0
                num_gpus = 1
        
        # Extract BS if present
        bs_match = re.search(r'_bs(\d+)_', filename)
        bs = int(bs_match.group(1)) if bs_match else None

        csv_data.append({
            'path': csv_path,
            'filename': filename,
            'timestamp': timestamp,
            'gpu_config': gpu_config,
            'rank': rank,
            'num_gpus': num_gpus,
            'bs': bs,
            'df': pd.read_csv(csv_path)
        })
        print(f"Loaded CSV file: {csv_path.name}")
    
    # Group CSV files: each group has all ranks for a configuration
    # single: 1 file, XnYt: X*Y files (all ranks)
    csv_groups = []
    
    # Sort by GPU config then rank
    csv_data.sort(key=lambda x: (x['gpu_config'], x['rank']))
    
    # Group by (gpu_config, bs)
   
    csv_groups = []
    
    # Helper for grouping key
    def get_group_key(item):
        # Treat None BS as -1 for sorting/grouping consistency if needed, but tuple works fine
        return (item['gpu_config'], item['bs'] if item['bs'] is not None else -1)

    # Sort data for grouping
    csv_data.sort(key=get_group_key)
    
    for (gpu_config, bs_val), group in itertools.groupby(csv_data, key=get_group_key):
        files = list(group)
        bs = files[0]['bs'] # restore original bs value (None or int)
        
        # Determine expected number of GPUs
        if gpu_config == 'single':
            expected = 1
        else:
            expected = files[0]['num_gpus']
            
        # Get ranks present
        # If we have multiple runs for same config/bs, we take the most recent file for each rank
        rank_files = {}
        for f in files:
            r = f['rank']
            if r not in rank_files or f['timestamp'] > rank_files[r]['timestamp']:
                rank_files[r] = f
        
        # Check if we have all ranks
        if len(rank_files) == expected and sorted(rank_files.keys()) == list(range(expected)):
            final_files = [rank_files[r] for r in sorted(rank_files.keys())]
            
            # Use max timestamp of the group as the group timestamp
            group_ts = max(f['timestamp'] for f in final_files)
            
            csv_groups.append({
                'timestamp': group_ts,
                'gpu_config': gpu_config,
                'bs': bs,
                'files': final_files
            })
        else:
            print(f"Warning: Incomplete ranks for config={gpu_config}, bs={bs}. Found: {sorted(rank_files.keys())}, Expected: {expected}")

    # Sort groups by:
    # 1. Number of GPUs (asc)
    # 2. Batch size (asc) - None considered as smallest (or use arbitrary order)
    def sort_key(g):
        num_gpus = 1 if g['gpu_config'] == 'single' else g['files'][0]['num_gpus']
        bs_val = g['bs'] if g['bs'] is not None else 0
        return (num_gpus, bs_val)
        
    csv_groups.sort(key=sort_key)
    
    print(f"\nGrouped into {len(csv_groups)} configuration(s)")
    for i, group in enumerate(csv_groups):
        bs_str = f", BS={group['bs']}" if group['bs'] is not None else ""
        print(f"  Group {i+1}: {group['gpu_config']}{bs_str} - {len(group['files'])} file(s) - Time: {group['timestamp']}")
    
    return csv_groups, output_path.name


def get_config_label(gpu_config):
    """
    Get descriptive label for GPU configuration.
    
    Parameters
    ----------
    gpu_config : str
        GPU configuration string (e.g., 'single', '1n2t', '1n4t')
        
    Returns
    -------
    str
        Descriptive label for the configuration
    """
    if gpu_config == 'single':
        return '1 GPU'
    else:
        # Extract number of GPUs from pattern like "1n2t" -> 1*2=2
        digits = re.findall(r'\d+', gpu_config)
        if len(digits) >= 2:
            num_gpus = int(digits[0]) * int(digits[1])
            return f'{num_gpus} GPUs - Distributed'
        return gpu_config


def get_group_label(group):
    """
    Get descriptive label for a group, including BS if present.
    """
    gpu_config = group['gpu_config']
    base_label = get_config_label(gpu_config)
    
    if group['bs'] is not None:
        return f"{base_label} (BS={group['bs']})"
    return base_label


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
    max_ranks = max(len(group['files']) for group in csv_groups)
    bar_width = 0.2
    # Ensure group spacing is large enough to hold the widest group plus padding
    group_spacing = max(1.0, max_ranks * bar_width * 1.25)
    
    x_tick_vals = []
    x_tick_text = []
    
    for group_idx, group in enumerate(csv_groups):
        config_label = get_group_label(group)
        files = sorted(group['files'], key=lambda x: x['rank'])
        num_ranks = len(files)
        
        # Center x-position for this group
        group_center_x = group_idx * group_spacing
        x_tick_vals.append(group_center_x)
        x_tick_text.append(config_label)
        
        for i, file_info in enumerate(files):
            df = file_info['df']
            rank = file_info['rank']
            
            # Aggregate metric value
            if metric_config['aggregation'] == 'mean':
                metric_value = df[metric_config['column']].mean()
            elif metric_config['aggregation'] == 'max':
                metric_value = df[metric_config['column']].max()
            else:
                raise ValueError(f"Unsupported aggregation: {metric_config['aggregation']}")
            
            if len(group['files']) > 1:
                label = f"{config_label} - Rank {rank}"
            else:
                label = config_label
            
            # Calculate manual bar position to center the group
            # Formula: center + (index - (count-1)/2) * width
            bar_x = group_center_x + (i - (num_ranks - 1) / 2) * bar_width
            
            fig.add_trace(go.Bar(
                name=label,
                x=[bar_x],
                y=[metric_value],
                width=[bar_width],
                text=[f"Rank {rank}" if len(group['files']) > 1 else ""],
                textposition='auto',
                hovertemplate=f'<b>{label}</b><br>' +
                              f"<b>{metric_config['hover_label']}:</b> %{{y:{metric_config['format']}}}<extra></extra>"
            ))
    
    fig.update_layout(
        title=dict(text=metric_config['title'], font=dict(size=38)),
        xaxis=dict(
            title='Configuration',
            tickmode='array',
            tickvals=x_tick_vals,
            ticktext=x_tick_text
        ),
        yaxis_title=metric_config['ylabel'],
        template='plotly_white',
        legend=dict(
            font=dict(size=28)
        ),
        font=dict(size=28)
    )
    
    output_path = Path(output_dir) / metric_config['filename']
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def visualize_rank_details(output_dir, results_dir='results_images'):
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
    plot_metric_by_rank(csv_groups, results_path, {
        'column': 'gradient_time_sec',
        'aggregation': 'mean',
        'title': 'Average Gradient Time by Rank',
        'ylabel': 'Average Gradient Time (seconds)',
        'hover_label': 'Avg Gradient Time',
        'format': '.3f',
        'filename': 'gpu_gradient_time_by_rank.html'
    })
    
    plot_metric_by_rank(csv_groups, results_path, {
        'column': 'denoise_time_sec',
        'aggregation': 'mean',
        'title': 'Average Denoiser Time by Rank',
        'ylabel': 'Average Denoiser Time (seconds)',
        'hover_label': 'Avg Denoiser Time',
        'format': '.3f',
        'filename': 'gpu_denoiser_time_by_rank.html'
    })
    
    plot_metric_by_rank(csv_groups, results_path, {
        'column': 'gradient_memory_peak_mb',
        'aggregation': 'max',
        'title': 'GPU Max Memory for Gradient by Rank',
        'ylabel': 'Max Memory Allocated (MB)',
        'hover_label': 'Max Gradient Memory',
        'format': '.2f',
        'filename': 'gpu_gradient_memory_by_rank.html'
    })
    
    plot_metric_by_rank(csv_groups, results_path, {
        'column': 'denoise_memory_peak_mb',
        'aggregation': 'max',
        'title': 'GPU Max Memory for Denoiser by Rank',
        'ylabel': 'Max Memory Allocated (MB)',
        'hover_label': 'Max Denoiser Memory',
        'format': '.2f',
        'filename': 'gpu_denoiser_memory_by_rank.html'
    })
    
    print("-" * 60)
    print("All visualizations completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize detailed per-rank GPU metrics from CSV files')
    parser.add_argument('output_dir', nargs='?', default='outputs/tomography_3d',
                        help='Path to the output directory containing CSV files')
    parser.add_argument('results_dir', nargs='?', default='results_images',
                        help='Directory to save visualization images')
    
    args = parser.parse_args()
    
    visualize_rank_details(args.output_dir, args.results_dir)
