"""GPU metrics tracking for performance benchmarking.

This module provides a GPUMetricsTracker class to monitor GPU memory and
execution time, with support for per-step breakdown of computational tasks
in iterative algorithms like Plug-and-Play optimization.

Tracks:
- GPU memory allocation (current, reserved, peak)
- Per-step execution time and memory deltas
"""
import time
import torch


class GPUMetricsTracker:
    """Tracks GPU metrics including memory usage and execution time.
    
    Designed for SLURM distributed environments where each process has
    its own assigned GPU(s). Supports per-step timing breakdown.
    
    Attributes
    ----------
    device : torch.device
        Target device (e.g., 'cuda:0' or 'cpu')
    has_cuda : bool
        Whether CUDA is available
    step_checkpoints : dict
        Stores timing checkpoints for each named step
    step_metrics : dict
        Aggregated metrics (time, memory deltas) for each step
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize GPU metrics tracker.
        
        Parameters
        ----------
        device : torch.device or str, optional
            Device to track. Default: CUDA if available, else CPU.
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.has_cuda = torch.cuda.is_available() and self.device.type == 'cuda'
        
        # Timing checkpoints: {step_name: {'start': time, 'end': time}}
        self.step_checkpoints = {}
        
        # Aggregated step metrics: {step_name: {'time': float, 'memory_delta_mb': float, ...}}
        self.step_metrics = {}
        
        # Peak memory tracking per step (before/after)
        self.step_memory_peaks = {}
    
    def snapshot(self, step_name, phase='start'):
        """Record a timing/memory checkpoint for a step.
        
        Call with phase='start' before a computational step, then
        phase='end' after it to automatically compute time delta and
        memory changes.
        
        Parameters
        ----------
        step_name : str
            Name of the computational step (e.g., 'gradient', 'denoise')
        phase : {'start', 'end'}, optional
            Phase of the step. Default: 'start'
        """
        if phase == 'start':
            # Reset peak memory stats for this step
            if self.has_cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            
            # Record start checkpoint
            if step_name not in self.step_checkpoints:
                self.step_checkpoints[step_name] = {}
            
            self.step_checkpoints[step_name]['start'] = time.perf_counter()
            self.step_checkpoints[step_name]['mem_start'] = (
                self._get_memory_allocated() if self.has_cuda else 0.0
            )
        
        elif phase == 'end':
            # Synchronize GPU operations before measuring
            if self.has_cuda:
                torch.cuda.synchronize(self.device)
            
            # Record end checkpoint
            end_time = time.perf_counter()
            mem_end = self._get_memory_allocated() if self.has_cuda else 0.0
            mem_peak = self._get_peak_memory_allocated() if self.has_cuda else 0.0
            
            if step_name not in self.step_checkpoints:
                self.step_checkpoints[step_name] = {}
            
            self.step_checkpoints[step_name]['end'] = end_time
            self.step_checkpoints[step_name]['mem_end'] = mem_end
            self.step_checkpoints[step_name]['mem_peak'] = mem_peak
            
            # Compute and store aggregated metrics
            self._compute_step_metrics(step_name)
        
        else:
            raise ValueError(f"Invalid phase: {phase}. Use 'start' or 'end'.")
    
    def _compute_step_metrics(self, step_name):
        """Compute aggregated metrics for a completed step.
        
        Parameters
        ----------
        step_name : str
            Name of the computational step
        """
        if step_name not in self.step_checkpoints:
            return
        
        checkpoint = self.step_checkpoints[step_name]
        
        if 'start' not in checkpoint or 'end' not in checkpoint:
            return
        
        # Compute time delta
        elapsed_time = checkpoint['end'] - checkpoint['start']
        
        # Compute memory deltas
        mem_start = checkpoint.get('mem_start', 0.0)
        mem_end = checkpoint.get('mem_end', 0.0)
        mem_peak = checkpoint.get('mem_peak', 0.0)
        
        memory_delta = mem_end - mem_start
        
        self.step_metrics[step_name] = {
            'time_sec': elapsed_time,
            'memory_allocated_mb': mem_end,
            'memory_delta_mb': memory_delta,
            'memory_peak_mb': mem_peak,
        }
    
    def get_step_metrics(self, step_name):
        """Retrieve aggregated metrics for a step.
        
        Parameters
        ----------
        step_name : str
            Name of the computational step
        
        Returns
        -------
        dict or None
            Dictionary with 'time_sec', 'memory_allocated_mb', 
            'memory_delta_mb', 'memory_peak_mb' keys.
            Returns None if step metrics not available.
        """
        return self.step_metrics.get(step_name, None)
    
    def get_all_step_metrics(self):
        """Retrieve all aggregated step metrics.
        
        Returns
        -------
        dict
            Dictionary mapping step names to their metrics dictionaries
        """
        return self.step_metrics.copy()
    
    def get_gpu_memory_snapshot(self):
        """Get current GPU memory state.
        
        Returns
        -------
        dict
            Dictionary with current memory statistics:
            - 'allocated_mb': currently allocated memory
            - 'reserved_mb': reserved memory from OS
            - 'max_allocated_mb': peak memory in session
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
        max_allocated = self._get_peak_memory_allocated()
        
        try:
            free, total = torch.cuda.mem_get_info(self.device)
            available = free / (1024 ** 2)
        except RuntimeError:
            available = 0.0
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allocated_mb': max_allocated,
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
    
    def reset_peak_memory(self):
        """Reset peak memory tracking.
        
        Useful for resetting between major phases of computation.
        """
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def reset(self):
        """Reset all tracked metrics and checkpoints."""
        self.step_checkpoints.clear()
        self.step_metrics.clear()
        self.step_memory_peaks.clear()
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
