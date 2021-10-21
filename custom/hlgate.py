import cirq;
import cirq_google
import numpy as np

"""
Hyahatiph Labs custom gate.
Defined by the phi ratio and 
Hadamard gate.
"""

"""Define a custom single-qubit gate."""
class HLGate(cirq.Gate):
    def __init__(self):
        super(HLGate, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return np.array([
            [1.0,  1.0],
            [1.0, -1.0]
        ]) / ((1 + np.sqrt(5)) / 2) # phi

    def _circuit_diagram_info_(self, args):
        return "HL"
