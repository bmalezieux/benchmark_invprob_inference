"""GPU metrics tracking for performance benchmarking.

This module provides a GPUMetricsTracker class to monitor GPU memory and
execution time, with support for per-step breakdown of computational tasks
in iterative algorithms like Plug-and-Play optimization.

Tracks:
- GPU memory allocation (current, reserved, peak)
- Per-step execution time and memory 
"""
import time
import torch
from contextlib import contextmanager


class GPUMetricsTracker:
    """Tracks GPU metrics including memory usage and execution time.
    
    Designed for SLURM distributed environments where each process has
    its own assigned GPU(s). Supports per-step timing breakdown.
    """
    
    def __init__(self, device=None):
        """Initialize GPU metrics tracker.
        
        Parameters
        ----------
        device : torch.device or str, optional
            Device to track. Default: CUDA if available, else CPU.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.has_cuda = torch.cuda.is_available() and self.device.type == 'cuda'
        
        # Step metrics: {step_name: {'time_sec': float, 'memory_delta_mb': float, ...}}
        self.step_metrics = {}
        
        # Initialize peak memory with current allocation
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.iteration_peak_memory_mb = self._get_peak_memory_allocated()
        else:
            self.iteration_peak_memory_mb = 0.0
    
    @contextmanager
    def track_step(self, step_name):
        """Context manager to track a computational step.
        
        Usage:
            with gpu_tracker.track_step('gradient'):
                # ... computational code ...
                pass
        
        Parameters
        ----------
        step_name : str
            Name of the computational step (e.g., 'gradient', 'denoise')
        """
        # Reset peak memory stats for this step
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        
        # Record start measurements
        start_time = time.perf_counter()
        mem_start = self._get_memory_allocated() if self.has_cuda else 0.0
        
        try:
            yield
        finally:
            # Synchronize GPU operations before measuring
            if self.has_cuda:
                torch.cuda.synchronize(self.device)
            
            # Record end measurements
            end_time = time.perf_counter()
            mem_end = self._get_memory_allocated() if self.has_cuda else 0.0
            mem_reserved = self._get_memory_reserved() if self.has_cuda else 0.0
            mem_peak = self._get_peak_memory_allocated() if self.has_cuda else 0.0
            
            # Compute and store metrics directly
            elapsed_time = end_time - start_time
            memory_delta = mem_end - mem_start
            
            # Update iteration peak memory with the maximum from this step
            if mem_peak > self.iteration_peak_memory_mb:
                self.iteration_peak_memory_mb = mem_peak
            
            self.step_metrics[step_name] = {
                'time_sec': elapsed_time,
                'memory_allocated_mb': mem_end,
                'memory_reserved_mb': mem_reserved,
                'memory_delta_mb': memory_delta,
                'memory_peak_mb': mem_peak,
            }
    
    def get_all_step_metrics(self):
        """Retrieve all aggregated step metrics.
        
        Returns
        -------
        dict
            Dictionary mapping step names to their metrics dictionaries
        """
        return self.step_metrics.copy()
    
    def reset_iteration_tracking(self):
        """Reset tracking for a new iteration.
        
        Call this at the start of each iteration to reset peak memory
        and clear step metrics.
        """
        # Reset step metrics
        self.step_metrics = {}
        
        # Reset peak memory tracking for new iteration
        if self.has_cuda:
            self.iteration_peak_memory_mb = self._get_peak_memory_allocated()
        else:
            self.iteration_peak_memory_mb = 0.0
    
    def get_gpu_memory_snapshot(self):
        """Get current GPU memory state.
        
        Returns
        -------
        dict
            Dictionary with current memory statistics:
            - 'allocated_mb': currently allocated memory
            - 'reserved_mb': reserved memory from OS
            - 'max_allocated_mb': peak memory for current iteration
            - 'available_mb': available memory
        """
        if not self.has_cuda:
            return {
                'allocated_mb': 0.0,
                'reserved_mb': 0.0,
                'max_allocated_mb': 0.0,
                'available_mb': 0.0,
            }
        
        allocated = self._get_memory_allocated()
        reserved = self._get_memory_reserved()
        
        try:
            free, total = torch.cuda.mem_get_info(self.device)
            available = free / (1024 ** 2)
        except RuntimeError:
            available = 0.0
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allocated_mb': self.iteration_peak_memory_mb,
            'available_mb': available,
        }
    
    def _get_memory_allocated(self):
        """Get currently allocated GPU memory in MB."""
        if not self.has_cuda:
            return 0.0
        return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
    
    def _get_memory_reserved(self):
        """Get reserved GPU memory in MB."""
        if not self.has_cuda:
            return 0.0
        return torch.cuda.memory_reserved(self.device) / (1024 ** 2)
    
    def _get_peak_memory_allocated(self):
        """Get peak allocated GPU memory in MB."""
        if not self.has_cuda:
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
    
    def capture_iteration_result(self):
        """Capture current iteration's metrics combining GPU snapshot and step metrics.
        
        Returns
        -------
        dict
            Dictionary with all GPU and per-step metrics for the iteration
        """
        # Get global GPU memory state
        result = self.get_gpu_memory_snapshot()
        
        # Rename keys to match expected format
        iteration_result = {
            'gpu_memory_allocated_mb': result['allocated_mb'],
            'gpu_memory_reserved_mb': result['reserved_mb'],
            'gpu_memory_max_allocated_mb': result['max_allocated_mb'],
            'gpu_available_memory_mb': result['available_mb'],
        }
        
        # Add per-step metrics
        for step_name, metrics in self.step_metrics.items():
            iteration_result[f'{step_name}_time_sec'] = metrics['time_sec']
            iteration_result[f'{step_name}_memory_allocated_mb'] = metrics['memory_allocated_mb']
            iteration_result[f'{step_name}_memory_reserved_mb'] = metrics['memory_reserved_mb']
            iteration_result[f'{step_name}_memory_delta_mb'] = metrics['memory_delta_mb']
            iteration_result[f'{step_name}_memory_peak_mb'] = metrics['memory_peak_mb']
        
        return iteration_result


def save_result_per_rank(all_results, name, max_batch_size=0):
    """Save GPU metrics per rank to CSV file.
    
    Parameters
    ----------
    all_results : list[dict]
        List of dictionaries containing GPU metrics for each iteration.
    name : str
        Base name for the output file (should include rank information).
    max_batch_size : int, optional
        Max batch size to include in filename if > 0. Default: 0.
    """
    if not all_results:
        return
        
    import pandas as pd
    from pathlib import Path
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with optional batch size suffix
    if max_batch_size > 0:
        csv_path = output_dir / f"{name}_bs{max_batch_size}_gpu_metrics.csv"
    else:
        csv_path = output_dir / f"{name}_gpu_metrics.csv"
    
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(all_results)} GPU metric records to {csv_path}")
