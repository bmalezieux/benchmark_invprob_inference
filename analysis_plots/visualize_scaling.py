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

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


def read_parquet_data(output_dir):
    """
    Read parquet file from the output directory.
    
    Parameters
    ----------
    output_dir : str or Path
        Path to the output directory containing benchmark results
        
    Returns
    -------
    parquet_df : pd.DataFrame
        DataFrame from parquet file
    result_name : str
        Name of the result (from output directory)
    """
    output_path = Path(output_dir)
    
    # Find and read parquet file
    parquet_files = list(output_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file found in {output_dir}")
    
    parquet_path = parquet_files[0]
    
    # Read parquet file with pandas
    parquet_df = pd.read_parquet(parquet_path)
    
    # Calculate total number of GPUs: nodes × gpus_per_node
    def get_total_gpus(row):
        nodes = row.get('p_solver_slurm_nodes', 1)
        gres = row.get('p_solver_slurm_gres', 'gpu:1')
        
        if isinstance(gres, str) and ':' in gres:
            gpus_per_node = int(gres.split(':')[1])
        else:
            gpus_per_node = 1
        
        return nodes * gpus_per_node
    
    def create_config_label(row):
        nodes = row.get('p_solver_slurm_nodes', 1)
        tasks = row.get('p_solver_slurm_ntasks_per_node', 1)
        
        if nodes == 1 and tasks == 1:
            return "Single GPU"
        elif nodes == 1:
            return f"1n{tasks}t"
        else:
            return f"{nodes}n{tasks}t"
    
    parquet_df['ngpu'] = parquet_df.apply(get_total_gpus, axis=1)
    parquet_df['config_label'] = parquet_df.apply(create_config_label, axis=1)
    
    # Get result name from output directory
    result_name = output_path.name
    
    # Detect the image size column name (could be 'p_dataset_image_size' or 'p_dataset_img_size')
    if 'p_dataset_image_size' in parquet_df.columns:
        img_size_col = 'p_dataset_image_size'
    elif 'p_dataset_img_size' in parquet_df.columns:
        img_size_col = 'p_dataset_img_size'
    else:
        raise KeyError("Could not find image size column (tried 'p_dataset_image_size' and 'p_dataset_img_size')")
    
    # Store the column name for later use
    parquet_df.attrs['img_size_col'] = img_size_col
    
    print(f"Loaded parquet file: {parquet_path.name}")
    print(f"Shape: {parquet_df.shape}")
    print(f"\nImage sizes found: {sorted(parquet_df[img_size_col].unique())}")
    print(f"\nGPU configurations found:")
    config_info = parquet_df.groupby(['config_label', 'ngpu']).size().reset_index(name='count')
    config_info = config_info.sort_values('ngpu')
    for _, row in config_info.iterrows():
        print(f"  - {row['config_label']:12s} ({row['ngpu']:2d} GPUs): {row['count']} data points")
    
    return parquet_df, result_name


def calculate_scaling_metrics(parquet_df):
    """
    Calculate strong scaling and parallel efficiency metrics per image size.
    
    Parameters
    ----------
    parquet_df : pd.DataFrame
        DataFrame containing benchmark results
        
    Returns
    -------
    scaling_dict : dict
        Dictionary with image size as key and DataFrame with scaling metrics as value
    """
    # Get the image size column name from DataFrame attributes
    img_size_col = parquet_df.attrs.get('img_size_col', 'p_dataset_image_size')
    
    # Get the maximum iteration reached for each GPU configuration
    max_iter_per_gpu = parquet_df.groupby('ngpu')['stop_val'].max()
    
    # Find the common maximum iteration across all GPU configs
    common_max_iter = max_iter_per_gpu.min()
    
    print(f"\nUsing iteration {common_max_iter} for scaling analysis")
    
    # Filter to only include data up to the common maximum iteration
    filtered_df = parquet_df[parquet_df['stop_val'] <= common_max_iter].copy()
    
    # Calculate metrics for each image size
    scaling_dict = {}
    
    for img_size in sorted(filtered_df[img_size_col].unique()):
        img_df = filtered_df[filtered_df[img_size_col] == img_size]
        
        # Calculate total time to reach each iteration for each GPU configuration
        scaling_data = []
        
        for ngpu in sorted(img_df['ngpu'].unique()):
            gpu_data = img_df[img_df['ngpu'] == ngpu]
            
            # Get time at the common max iteration
            max_iter_data = gpu_data[gpu_data['stop_val'] == common_max_iter]
            if len(max_iter_data) > 0:
                total_time = max_iter_data['time'].values[0]
                config_label = max_iter_data['config_label'].values[0]
                gradient_time = max_iter_data['objective_gradient_time_sec'].values[0]
                denoiser_time = max_iter_data['objective_denoise_time_sec'].values[0]
                
                scaling_data.append({
                    'ngpu': ngpu,
                    'config_label': config_label,
                    'total_time': total_time,
                    'gradient_time': gradient_time,
                    'denoiser_time': denoiser_time,
                    'iteration': common_max_iter
                })
        
        if len(scaling_data) > 0:
            scaling_df = pd.DataFrame(scaling_data)
            
            # Calculate speedup and efficiency
            baseline_time = scaling_df[scaling_df['ngpu'] == scaling_df['ngpu'].min()]['total_time'].values[0]
            baseline_gradient_time = scaling_df[scaling_df['ngpu'] == scaling_df['ngpu'].min()]['gradient_time'].values[0]
            baseline_denoiser_time = scaling_df[scaling_df['ngpu'] == scaling_df['ngpu'].min()]['denoiser_time'].values[0]
            baseline_ngpu = scaling_df['ngpu'].min()
            
            scaling_df['speedup'] = baseline_time / scaling_df['total_time']
            scaling_df['gradient_speedup'] = baseline_gradient_time / scaling_df['gradient_time']
            scaling_df['denoiser_speedup'] = baseline_denoiser_time / scaling_df['denoiser_time']
            scaling_df['ideal_speedup'] = scaling_df['ngpu'] / baseline_ngpu
            scaling_df['efficiency'] = (scaling_df['speedup'] / scaling_df['ideal_speedup']) * 100
            
            scaling_dict[img_size] = scaling_df
    
    return scaling_dict


def plot_scaling_metric(scaling_dict, output_dir, metric_name, metric_col, title, y_label, y_range=None, text_format='{:.2f}x', show_legend=True):
    """
    Generic function to plot scaling metrics vs number of GPUs for each image size.
    
    Parameters
    ----------
    scaling_dict : dict
        Dictionary with image size as key and DataFrame with scaling metrics as value
    output_dir : str or Path
        Directory to save the plot
    metric_name : str
        Name of the metric (used for filename)
    metric_col : str
        Column name in DataFrame to plot
    title : str
        Title of the plot
    y_label : str
        Y-axis label
    y_range : list, optional
        Y-axis range [min, max]
    text_format : str
        Format string for text labels (e.g., '{:.2f}x' or '{:.1f}%')
    show_legend : bool
        Whether to show legend (default: True)
    """
    fig = go.Figure()
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for idx, (img_size, scaling_df) in enumerate(sorted(scaling_dict.items())):
        color = colors[idx % len(colors)]
        
        # Format text labels
        text_values = scaling_df[metric_col].round(2 if 'speedup' in metric_col else 1)
        text_labels = [f'{val:.2f}x' if 'speedup' in metric_col else f'{val:.1f}%' for val in text_values]
        
        # Plot actual metric
        fig.add_trace(go.Scatter(
            x=scaling_df['ngpu'],
            y=scaling_df[metric_col],
            mode='lines+markers+text',
            name=f'Image {img_size}×{img_size}',
            line=dict(width=4, color=color),
            marker=dict(size=12),
            text=text_labels,
            texttemplate='%{text}',
            textposition='top center',
            textfont=dict(size=14, color=color),
            hovertemplate='<b>Image size:</b> ' + str(img_size) + '<br>' +
                          '<b>Config:</b> ' + scaling_df['config_label'] + '<br>' +
                          '<b>GPUs:</b> %{x}<br>' +
                          f'<b>{y_label}:</b> %{{y:.2f}}<br>' +
                          '<extra></extra>'
        ))
    
    # Add ideal reference line if metric contains 'speedup' or is 'efficiency'
    all_ngpus = sorted(set([ngpu for scaling_df in scaling_dict.values() for ngpu in scaling_df['ngpu']]))
    
    if 'speedup' in metric_col or metric_col == 'efficiency':
        if metric_col == 'efficiency':
            # For efficiency, ideal is 100%
            ideal_values = [100] * len(all_ngpus)
            ideal_name = 'Ideal Efficiency (100%)'
        else:
            # For speedup, ideal is linear
            baseline_ngpu = min(all_ngpus)
            ideal_values = [ngpu / baseline_ngpu for ngpu in all_ngpus]
            ideal_name = 'Ideal Speedup'
        
        # Add ideal reference line (no text, no markers)
        fig.add_trace(go.Scatter(
            x=all_ngpus,
            y=ideal_values,
            mode='lines',
            name=ideal_name,
            line=dict(width=3, color='black', dash='dash'),
            hovertemplate='<b>Ideal:</b> %{y:.2f}<extra></extra>' if metric_col != 'efficiency' else '<b>Ideal:</b> 100%<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis=dict(
            title='Number of GPUs',
            tickmode='array',
            tickvals=all_ngpus,
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=y_label,
            range=y_range,
            tickfont=dict(size=14)
        ),
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=50, r=20, t=60, b=50),
        showlegend=show_legend,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(size=12)
        ) if show_legend else None,
        font=dict(size=14)
    )
    
    output_path = Path(output_dir) / f'{metric_name}.html'
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")
    
    return fig


def create_dashboard(results_path, result_name):
    """Create an HTML dashboard combining all visualizations in a 2x2 grid."""
    from datetime import datetime
    
    required_files = ['strong_scaling_speedup.html', 'strong_scaling_gradient_speedup.html',
                     'strong_scaling_denoiser_speedup.html', 'parallel_efficiency.html']
    
    for file in required_files:
        if not (results_path / file).exists():
            print(f"Warning: Required file not found: {results_path / file}")
            return None
    
    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scaling Results Dashboard - {result_name}</title>
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
            <h1>📊 Scaling Results Dashboard</h1>
            <p>{result_name.replace('_', ' ').title()}</p>
        </div>
        <div class="timestamp">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</div>
        <div class="dashboard-grid">
            <div class="chart-container"><iframe src="strong_scaling_speedup.html" title="Strong Scaling Speedup"></iframe></div>
            <div class="chart-container"><iframe src="parallel_efficiency.html" title="Parallel Efficiency"></iframe></div>
            <div class="chart-container"><iframe src="strong_scaling_gradient_speedup.html" title="Gradient Speedup"></iframe></div>
            <div class="chart-container"><iframe src="strong_scaling_denoiser_speedup.html" title="Denoiser Speedup"></iframe></div>
        </div>
        <div class="footer">Scaling Results Dashboard </div>
    </div>
</body>
</html>"""
    
    dashboard_path = results_path / 'dashboard_scaling.html'
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    
    print(f"Dashboard created: {dashboard_path}")
    return dashboard_path


def visualize_scaling(output_dir, results_dir='results_images'):
    """
    Main function to create all scaling visualizations.
    
    Parameters
    ----------
    output_dir : str or Path
        Path to the output directory containing benchmark results
    results_dir : str or Path
        Directory to save visualization images (default: 'results_images')
    """
    print(f"Reading data from: {output_dir}")
    print("-" * 60)
    
    # Read parquet data
    parquet_df, result_name = read_parquet_data(output_dir)
    
    # Calculate scaling metrics
    print("-" * 60)
    print("Calculating scaling metrics...")
    scaling_dict = calculate_scaling_metrics(parquet_df)
    
    if len(scaling_dict) == 0:
        print("Error: No data available for scaling analysis")
        return
    
    # Display scaling summary
    print("\nScaling Summary:")
    for img_size, scaling_df in sorted(scaling_dict.items()):
        print(f"\nImage size {img_size}×{img_size}:")
        print(scaling_df[['config_label', 'ngpu', 'total_time', 'speedup', 'efficiency']].to_string(index=False))
    
    # Create results directory with scaling prefix
    results_path = Path(results_dir) / f"scaling_{result_name}"
    results_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\nSaving visualizations to: {results_path}")
    print("-" * 60)
    print("Creating visualizations...")
    print("-" * 60)
    
    # Create plots
    plot_scaling_metric(
        scaling_dict, results_path,
        'strong_scaling_speedup',
        'speedup',
        'Strong Scaling: Speedup vs Number of GPUs',
        'Speedup',
        show_legend=True
    )
    
    plot_scaling_metric(
        scaling_dict, results_path,
        'strong_scaling_gradient_speedup',
        'gradient_speedup',
        'Strong Scaling: Gradient Speedup vs Number of GPUs',
        'Gradient Speedup',
        show_legend=False
    )
    
    plot_scaling_metric(
        scaling_dict, results_path,
        'strong_scaling_denoiser_speedup',
        'denoiser_speedup',
        'Strong Scaling: Denoiser Speedup vs Number of GPUs',
        'Denoiser Speedup',
        show_legend=False
    )
    
    plot_scaling_metric(
        scaling_dict, results_path,
        'parallel_efficiency',
        'efficiency',
        'Parallel Efficiency vs Number of GPUs',
        'Parallel Efficiency (%)',
        y_range=[0, 110],
        show_legend=True
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
    
    parser = argparse.ArgumentParser(description='Visualize scaling benchmark results from parquet file')
    parser.add_argument('output_dir', nargs='?', default='outputs/scaling/highres_color_image',
                        help='Path to the output directory containing benchmark results')
    parser.add_argument('results_dir', nargs='?', default='results_images',
                        help='Directory to save visualization images')
    
    args = parser.parse_args()
    
    visualize_scaling(args.output_dir, args.results_dir)
