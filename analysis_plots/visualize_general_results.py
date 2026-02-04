"""
Visualization script for general benchmark performance results.

This script reads parquet files from benchmark outputs and creates
interactive Plotly visualizations for overall performance metrics.

The script generates the following visualizations:
1. PSNR vs Time - Shows how PSNR evolves over computation time
2. PSNR vs Iteration - Shows PSNR convergence over iterations
3. Time Breakdown - Stacked bar chart showing average time spent in gradient and denoiser operations
4. GPU Memory Usage - Bar chart showing average GPU memory for gradient and denoiser

Usage
-----
From command line:
    python visualize_general_results.py <output_dir> [results_dir]

Arguments:
    output_dir : Path to the output directory containing benchmark results (parquet file)
    results_dir : Directory to save visualization images (default: 'results_images')

Examples
--------

# Custom output directory
python analysis_plots/visualize_general_results.py outputs/tomography_2d 

Output Structure
----------------
Visualizations are saved in: results_dir/result_name/
where result_name is the name of the output directory (e.g., 'highres_color_image')

"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


def create_config_label(row):
    """
    Create a descriptive label for a configuration based on key parameters.
    
    Parameters
    ----------
    row : pd.Series
        Row from the parquet dataframe
        
    Returns
    -------
    str
        Descriptive label for the configuration
    """
    # Extract number of GPUs from slurm_gres (e.g., 'gpu:1' -> 1, 'gpu:2' -> 2)
    gres = row.get('p_solver_slurm_gres', 'gpu:1')
    if isinstance(gres, str) and ':' in gres:
        ngpu = int(gres.split(':')[1])
    else:
        ngpu = 1
    
    dist_phy = row.get('p_solver_distribute_physics', False)
    dist_den = row.get('p_solver_distribute_denoiser', False)
    patch = row.get('p_solver_patch_size', 0)
    max_batch = row.get('p_solver_max_batch_size', None)
    
    # Build descriptive label based on key differences
    if max_batch is not None and max_batch > 0:
        if ngpu == 1:
            label = f"1 GPU - Batch={max_batch}"
        else:
            label = f"{ngpu} GPUs - Batch={max_batch}"
    else:
        if ngpu == 1:
            label = "1 GPU"
        else:
            label = f"{ngpu} GPUs"
        
        # Add distribution info
        if dist_phy and dist_den:
            label += " - Distributed"
        elif dist_phy:
            label += " - Dist Physics"
        elif dist_den:
            label += " - Dist Denoiser"
    
    return label


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
        DataFrame from parquet file with config labels
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
    
    # Extract sorting keys
    def get_ngpu(row):
        gres = row.get('p_solver_slurm_gres', 'gpu:1')
        if isinstance(gres, str) and ':' in gres:
            return int(gres.split(':')[1])
        return 1
    
    parquet_df['ngpu'] = parquet_df.apply(get_ngpu, axis=1)
    parquet_df['batch'] = parquet_df['p_solver_max_batch_size'].fillna(0)
    
    # Create descriptive labels based on configuration differences
    parquet_df['config_label'] = parquet_df.apply(create_config_label, axis=1)
    
    # Sort dataframe by ngpu then batch
    parquet_df.sort_values(by=['ngpu', 'batch'], inplace=True)
    
    # Get result name from output directory
    result_name = output_path.name
    
    print(f"Loaded parquet file: {parquet_path.name}")
    print(f"Shape: {parquet_df.shape}")
    print(f"\nConfigurations found:")
    for label in parquet_df['config_label'].unique():
        count = (parquet_df['config_label'] == label).sum()
        print(f"  - {label}: {count-1} iterations")
    
    return parquet_df, result_name


def plot_psnr_vs_time(parquet_df, output_dir):
    """
    Plot PSNR (loss function) vs time for each job.
    
    Parameters
    ----------
    parquet_df : pd.DataFrame
        DataFrame containing benchmark results
    output_dir : str or Path
        Directory to save the plot
    """
    fig = go.Figure()
    
    # Group by configuration label
    groups = parquet_df.groupby('config_label')
    
    # Get sorted labels
    sorted_labels = parquet_df[['config_label', 'ngpu', 'batch']].drop_duplicates().sort_values(by=['ngpu', 'batch'])['config_label']
    
    for label in sorted_labels:
        group = groups.get_group(label)
        
        # Sort by time
        group = group.sort_values('time')
        
        fig.add_trace(go.Scatter(
            x=group['time'],
            y=group['objective_psnr'],
            mode='lines+markers',
            name=label,
            line=dict(width=4),
            marker=dict(size=10),
            hovertemplate='<b>Config:</b> ' + label + '<br>' +
                          '<b>Time:</b> %{x:.3f}s<br>' +
                          '<b>PSNR:</b> %{y:.2f} dB<br>' +
                          '<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='PSNR vs Time', font=dict(size=38)),
        xaxis_title='Time (seconds)',
        yaxis_title='PSNR (dB)',
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            font=dict(size=28)
        ),
        font=dict(size=28)
    )
    
    output_path = Path(output_dir) / 'psnr_vs_time.html'
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")
    
    return fig


def plot_psnr_vs_iteration(parquet_df, output_dir):
    """
    Plot PSNR (loss function) vs iteration (stop_val).
    
    Parameters
    ----------
    parquet_df : pd.DataFrame
        DataFrame containing benchmark results
    output_dir : str or Path
        Directory to save the plot
    """
    fig = go.Figure()
    
    # Group by configuration label
    groups = parquet_df.groupby('config_label')
    
    # Get sorted labels
    sorted_labels = parquet_df[['config_label', 'ngpu', 'batch']].drop_duplicates().sort_values(by=['ngpu', 'batch'])['config_label']
    
    for label in sorted_labels:
        group = groups.get_group(label)
        
        # Sort by iteration
        group = group.sort_values('stop_val')
        
        fig.add_trace(go.Scatter(
            x=group['stop_val'],
            y=group['objective_psnr'],
            mode='lines+markers',
            name=label,
            line=dict(width=4),
            marker=dict(size=10),
            hovertemplate='<b>Config:</b> ' + label + '<br>' +
                          '<b>Iteration:</b> %{x}<br>' +
                          '<b>PSNR:</b> %{y:.2f} dB<br>' +
                          '<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='PSNR vs Iteration', font=dict(size=38)),
        xaxis_title='Iteration',
        yaxis_title='PSNR (dB)',
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            font=dict(size=34)
        ),
        font=dict(size=28)
    )
    
    output_path = Path(output_dir) / 'psnr_vs_iteration.html'
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")
    
    return fig

def plot_time_breakdown_stacked(parquet_df, output_dir):
    """
    Plot stacked bar chart showing average time for gradient and denoiser.
    
    Parameters
    ----------
    parquet_df : pd.DataFrame
        DataFrame containing benchmark results
    output_dir : str or Path
        Directory to save the plot
    """
    # Aggregate data by configuration
    agg_data = parquet_df.groupby('config_label').agg({
        'objective_gradient_time_sec': 'mean',
        'objective_denoise_time_sec': 'mean',
        'ngpu': 'first',
        'batch': 'first'
    }).reset_index()
    
    agg_data.rename(columns={'config_label': 'label'}, inplace=True)
    
    # Calculate total time and percentages
    agg_data['total_time'] = agg_data['objective_gradient_time_sec'] + agg_data['objective_denoise_time_sec']
    agg_data['denoiser_pct'] = (agg_data['objective_denoise_time_sec'] / agg_data['total_time']) * 100
    
    # Sort by ngpu then batch
    agg_data.sort_values(by=['ngpu', 'batch'], inplace=True)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Gradient',
        x=agg_data['label'],
        y=agg_data['objective_gradient_time_sec'],
        marker_color='blue',
        width=0.5,
        hovertemplate='<b>Gradient Time:</b> %{y:.3f}s<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Denoiser',
        x=agg_data['label'],
        y=agg_data['objective_denoise_time_sec'],
        marker_color='orange',
        text=agg_data['denoiser_pct'],
        texttemplate='%{text:.1f}%',
        textposition='auto',
        textangle=0,
        textfont=dict(size=28),
        width=0.5,
        hovertemplate='<b>Denoiser Time:</b> %{y:.3f}s (%{text:.1f}%)<extra></extra>'
    ))
    
    # Add Total Time annotation at the top of each stack
    fig.add_trace(go.Scatter(
        x=agg_data['label'],
        y=agg_data['total_time'],
        mode='text',
        text=agg_data['total_time'],
        texttemplate='%{text:.1f}s',
        textposition='top center',
        textfont=dict(size=28),
        cliponaxis=False,
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text='Average Time Breakdown per Step', font=dict(size=38)),
        xaxis_title='Configuration',
        yaxis=dict(title='Average Time (seconds)', range=[0, agg_data['total_time'].max() * 1.2]),
        xaxis=dict(tickfont=dict(size=28) ),
        barmode='stack',
        template='plotly_white',
        margin=dict(t=120),
        legend=dict(
            font=dict(size=34)
        ),
        hovermode='x unified',
        font=dict(size=28)
    )
    
    output_path = Path(output_dir) / 'time_breakdown_stacked.html'
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")
    
    return fig


def plot_gpu_memory_from_parquet(parquet_df, output_dir):
    """
    Plot GPU max memory allocated from parquet file with gradient/denoiser breakdown.
    
    Parameters
    ----------
    parquet_df : pd.DataFrame
        DataFrame containing benchmark results
    output_dir : str or Path
        Directory to save the plot
    """
    # Aggregate data by configuration
    agg_data = parquet_df.groupby('config_label').agg({
        'objective_gradient_memory_peak_mb': 'mean',
        'objective_denoise_memory_peak_mb': 'mean',
        'ngpu': 'first',
        'batch': 'first'
    }).reset_index()
    
    agg_data.rename(columns={'config_label': 'label'}, inplace=True)
    
    # Sort by ngpu then batch
    agg_data.sort_values(by=['ngpu', 'batch'], inplace=True)
    
    # Create grouped bar chart (gradient and denoiser side by side)
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Gradient',
        x=agg_data['label'],
        y=agg_data['objective_gradient_memory_peak_mb'],
        marker_color='blue',
        text=agg_data['objective_gradient_memory_peak_mb'],
        texttemplate='%{y:.1f} MB',
        textposition='auto',
        textangle=0,
        textfont=dict(size=24),
        hovertemplate='<b>Gradient Memory:</b> %{y:.2f} MB<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Denoiser',
        x=agg_data['label'],
        y=agg_data['objective_denoise_memory_peak_mb'],
        marker_color='orange',
        text=agg_data['objective_denoise_memory_peak_mb'],
        texttemplate='%{y:.1f} MB',
        textposition='auto',
        textangle=0,
        textfont=dict(size=20),
        hovertemplate='<b>Denoiser Memory:</b> %{y:.2f} MB<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Average GPU Max Memory Allocated (Gradient vs Denoiser)', font=dict(size=38)),
        xaxis_title='Configuration',
        yaxis_title='Max Memory Allocated (MB)',
        xaxis=dict(tickfont=dict(size=24) ),
        barmode='group',
        template='plotly_white',
        legend=dict(
            font=dict(size=24)
        ),
        font=dict(size=24)
    )
    
    output_path = Path(output_dir) / 'gpu_memory_gradient_denoiser.html'
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")
    
    return fig


def visualize_general_results(output_dir, results_dir='results_images'):
    """
    Main function to create all general performance visualizations.
    
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
    
    # Create results directory with result name subdirectory
    results_path = Path(results_dir) / result_name
    results_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Saving visualizations to: {results_path}")
    print("-" * 60)
    print("Creating visualizations...")
    print("-" * 60)
    
    # Create plots from parquet data
    plot_psnr_vs_time(parquet_df, results_path)
    plot_psnr_vs_iteration(parquet_df, results_path)
    plot_time_breakdown_stacked(parquet_df, results_path)
    plot_gpu_memory_from_parquet(parquet_df, results_path)
    
    print("-" * 60)
    print("All visualizations completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize general benchmark results from parquet file')
    parser.add_argument('output_dir', nargs='?', default='outputs/highres_color_image',
                        help='Path to the output directory containing benchmark results')
    parser.add_argument('results_dir', nargs='?', default='results_images',
                        help='Directory to save visualization images')
    
    args = parser.parse_args()
    
    visualize_general_results(args.output_dir, args.results_dir)
