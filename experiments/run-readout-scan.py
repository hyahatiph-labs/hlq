# Put in a file named run-readout-scan.py

import datetime
import cirq_google as cg
import recirq.readout_scan.tasks as rst

MAX_N_QUBITS = 5

def main():
    """Main driver script entry point.

    This function contains configuration options and you will likely need
    to edit it to suit your needs. Of particular note, please make sure
    `dataset_id` and `device_name`
    are set how you want them. You may also want to change the values in
    the list comprehension to set the qubits.
    """
    # Uncomment below for an auto-generated unique dataset_id
    # dataset_id = datetime.datetime.now().isoformat(timespec='minutes')
    dataset_id = '2020-02-tutorial'
    data_collection_tasks = [
        rst.ReadoutScanTask(
            dataset_id=dataset_id,
            device_name='Syc23-simulator',
            n_shots=40_000,
            qubit=qubit,
            resolution_factor=6,
        )
        for qubit in cg.Sycamore23.qubits[:MAX_N_QUBITS]
    ]

    for dc_task in data_collection_tasks:
        rst.run_readout_scan(dc_task)


if __name__ == '__main__':
    main()
