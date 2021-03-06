import os

import numpy as np
import sympy

import cirq
import recirq

@recirq.json_serializable_dataclass(namespace='recirq.readout_scan', 
                                    registry=recirq.Registry,
                                    frozen=True)
class ReadoutScanTask:
    """Scan over Ry(theta) angles from -pi/2 to 3pi/2 tracing out a sinusoid
    which is primarily affected by readout error.

    See Also:
        :py:func:`run_readout_scan`

    Attributes:
        dataset_id: A unique identifier for this dataset.
        device_name: The device to run on, by name.
        n_shots: The number of repetitions for each theta value.
        qubit: The qubit to benchmark.
        resolution_factor: We select the number of points in the linspace
            so that the special points: (-1/2, 0, 1/2, 1, 3/2) * pi are
            always included. The total number of theta evaluations
            is resolution_factor * 4 + 1.
    """
    dataset_id: str
    device_name: str
    n_shots: int
    qubit: cirq.GridQubit
    resolution_factor: int

    @property
    def fn(self):
        n_shots = _abbrev_n_shots(n_shots=self.n_shots)
        qubit = _abbrev_grid_qubit(self.qubit)
        return (f'{self.dataset_id}/'
                f'{self.device_name}/'
                f'q-{qubit}/'
                f'ry_scan_{self.resolution_factor}_{n_shots}')


# Define the following helper functions to make nicer `fn` keys
# for the tasks:

def _abbrev_n_shots(n_shots: int) -> str:
    """Shorter n_shots component of a filename"""
    if n_shots % 1000 == 0:
        return f'{n_shots // 1000}k'
    return str(n_shots)

def _abbrev_grid_qubit(qubit: cirq.GridQubit) -> str:
    """Formatted grid_qubit component of a filename"""
    return f'{qubit.row}_{qubit.col}'

EXPERIMENT_NAME = 'readout-scan'
DEFAULT_BASE_DIR = os.path.expanduser(f'~/cirq-results/{EXPERIMENT_NAME}')

def run_readout_scan(task: ReadoutScanTask,
                     base_dir=None):
    """Execute a :py:class:`ReadoutScanTask` task."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if recirq.exists(task, base_dir=base_dir):
        print(f"{task} already exists. Skipping.")
        return

    # Create a simple circuit
    theta = sympy.Symbol('theta')
    circuit = cirq.Circuit([
        cirq.ry(theta).on(task.qubit),
        cirq.measure(task.qubit, key='z')
    ])

    # Use utilities to map sampler names to Sampler objects
    sampler = recirq.get_sampler_by_name(device_name=task.device_name)

    # Use a sweep over theta values.
    # Set up limits so we include (-1/2, 0, 1/2, 1, 3/2) * pi
    # The total number of points is resolution_factor * 4 + 1
    n_special_points: int = 5
    resolution_factor = task.resolution_factor
    theta_sweep = cirq.Linspace(theta, -np.pi / 2, 3 * np.pi / 2,
                                resolution_factor * (n_special_points - 1) + 1)
    thetas = np.asarray([v for ((k, v),) in theta_sweep.param_tuples()])
    flat_circuit, flat_sweep = cirq.flatten_with_sweep(circuit, theta_sweep)

    # Run the jobs
    print(f"Collecting data for {task.qubit}", flush=True)
    results = sampler.run_sweep(program=flat_circuit, params=flat_sweep,
                                repetitions=task.n_shots)

    # Save the results
    recirq.save(task=task, data={
        'thetas': thetas,
        'all_bitstrings': [
            recirq.BitArray(np.asarray(r.measurements['z']))
            for r in results]
    }, base_dir=base_dir)

