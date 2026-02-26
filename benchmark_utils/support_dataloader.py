import torch
from deepinv.utils.tensorlist import TensorList


def collate_deepinv_batch(batch):
    """Custom collate function to handle TensorList objects from deepinv.

    This function properly handles batching of ground truths and measurements,
    including special treatment for TensorList objects from stacked physics operators.

    Parameters
    ----------
    batch : list of tuples
        List of (ground_truth, measurement) pairs from dataset.

    Returns
    -------
    tuple
        (ground_truth_batch, measurement_batch) where:
        - ground_truth_batch: torch.Tensor with batch dimension
        - measurement_batch: torch.Tensor, TensorList, or list with batch dimension
    """
    if len(batch) == 1:
        ground_truth, measurement = batch[0]
        # Add batch dimension if needed
        if ground_truth.ndim == 3:
            ground_truth = ground_truth.unsqueeze(0)
        if isinstance(measurement, TensorList):
            measurement = TensorList(
                [m.unsqueeze(0) if m.ndim == 3 else m for m in measurement]
            )
        elif isinstance(measurement, (list, tuple)):
            measurement = [m.unsqueeze(0) if m.ndim == 3 else m for m in measurement]
        elif measurement.ndim == 3:
            measurement = measurement.unsqueeze(0)
        return ground_truth, measurement
    else:
        # For batch_size > 1, stack ground truths and handle measurements
        ground_truths = []
        measurements = []
        for gt, meas in batch:
            ground_truths.append(gt)
            measurements.append(meas)

        # Stack ground truths
        ground_truth_batch = torch.stack(ground_truths, dim=0)

        # Handle measurements - if TensorList, stack each operator's measurements
        if isinstance(measurements[0], TensorList):
            # Stack measurements for each operator separately
            num_operators = len(measurements[0])
            stacked_measurements = []
            for op_idx in range(num_operators):
                op_measurements = [meas[op_idx] for meas in measurements]
                stacked_measurements.append(torch.stack(op_measurements, dim=0))
            measurement_batch = TensorList(stacked_measurements)
        elif isinstance(measurements[0], (list, tuple)):
            # Stack list/tuple measurements
            num_operators = len(measurements[0])
            stacked_measurements = []
            for op_idx in range(num_operators):
                op_measurements = [meas[op_idx] for meas in measurements]
                stacked_measurements.append(torch.stack(op_measurements, dim=0))
            measurement_batch = stacked_measurements
        else:
            # Single tensor measurements
            measurement_batch = torch.stack(measurements, dim=0)

        return ground_truth_batch, measurement_batch


class ClampedHDF5Dataset:
    """Wrapper for HDF5Dataset that clamps measurements to [min_val, max_val] range.

    Clamping is performed once during initialization and cached for efficiency.

    Parameters
    ----------
    hdf5_dataset : HDF5Dataset
        The underlying HDF5 dataset to wrap.
    min_val : float, optional
        Minimum value for clamping. Default: 0.0
    max_val : float, optional
        Maximum value for clamping. Default: 1.0
    """

    def __init__(self, hdf5_dataset, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

        # Load and clamp all data once at initialization
        print(f"Clamping {len(hdf5_dataset)} measurements to [{min_val}, {max_val}]...")
        self.data = []
        for i in range(len(hdf5_dataset)):
            ground_truth, measurement = hdf5_dataset[i]

            # Clamp measurements to valid pixel range
            if isinstance(measurement, TensorList):
                measurement = TensorList(
                    [torch.clamp(m, min_val, max_val) for m in measurement]
                )
            elif isinstance(measurement, (list, tuple)):
                measurement = [torch.clamp(m, min_val, max_val) for m in measurement]
            else:
                measurement = torch.clamp(measurement, min_val, max_val)

            self.data.append((ground_truth, measurement))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
