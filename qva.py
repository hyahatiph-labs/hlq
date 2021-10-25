""" Quantum Variational Algorithm """
import cirq
import random
import numpy as np
import sympy
from custom.hlgate import HLGate

hlg = HLGate()

# define the length and width of the grid.
length = 3

# define qubits on the grid.
qubits = cirq.GridQubit.square(length)

print("Defined qubits:")
print(qubits)
print("\n")

# apply custom gate on even X gate on add
circuit = cirq.Circuit()
circuit.append(hlg(q) for q in qubits if (q.row + q.col) % 2 == 0)
circuit.append(cirq.X(q) for q in qubits if (q.row + q.col) % 2 == 1)

print("Circuit:")
print(circuit)
print("\n")

""" Create the Ansatz """
def rot_x_layer(length, half_turns):
    """Yields X rotations by half_turns on a square grid of given length."""

    # Define the gate once and then re-use it for each Operation.
    rot = cirq.XPowGate(exponent=half_turns)

    # Create an X rotation Operation for each qubit in the grid.
    for i in range(length):
        for j in range(length):
            yield rot(cirq.GridQubit(i, j))

# Create the circuit using the rot_x_layer generator
circuit = cirq.Circuit()
circuit.append(rot_x_layer(2, 0.1))
print("Ansatz:")
print(circuit)
print("\n")

""" Generate random problem instances """

def rand2d(rows, cols):
    return [[random.choice([+1, -1]) for _ in range(cols)] for _ in range(rows)]

def random_instance(length):
    # transverse field terms
    h = rand2d(length, length)
    # links within a row
    jr = rand2d(length - 1, length)
    # links within a column
    jc = rand2d(length, length - 1)
    return (h, jr, jc)

h, jr, jc = random_instance(3)
print("Random instances:")
print(f'transverse fields: {h}')
print(f'row j fields: {jr}')
print(f'column j fields: {jc}')
print("\n")

""" 
Apply an initial mixing step that puts all 
the qubits into the |+> = 1/sqrt(2)(|0> + |1>)state.
"""

def prepare_plus_layer(length):
    for i in range(length):
        for j in range(length):
            yield cirq.H(cirq.GridQubit(i, j))

""" 
Apply a cirq.ZPowGate for the same parameterfor all 
qubits where the transverse field term h is +1.
"""

def rot_z_layer(h, half_turns):
    """Yields Z rotations by half_turns conditioned on the field h."""
    gate = cirq.ZPowGate(exponent=half_turns)
    for i, h_row in enumerate(h):
        for j, h_ij in enumerate(h_row):
            if h_ij == 1:
                yield gate(cirq.GridQubit(i, j))

"""
Apply a cirq.CZPowGate for the same parameter between all 
qubits where the coupling field term J is +1. If the field is -1, 
apply cirq.CZPowGate conjugated by X gates on all qubits.
"""

def rot_11_layer(jr, jc, half_turns):
    """Yields rotations about |11> conditioned on the jr and jc fields."""
    cz_gate = cirq.CZPowGate(exponent=half_turns)    
    for i, jr_row in enumerate(jr):
        for j, jr_ij in enumerate(jr_row):
            q = cirq.GridQubit(i, j)
            q_1 = cirq.GridQubit(i + 1, j)
            if jr_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)
            yield cz_gate(q, q_1)
            if jr_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)

    for i, jc_row in enumerate(jc):
        for j, jc_ij in enumerate(jc_row):
            q = cirq.GridQubit(i, j)
            q_1 = cirq.GridQubit(i, j + 1)
            if jc_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)
            yield cz_gate(q, q_1)
            if jc_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)

"""
Apply an cirq.XPowGate for the same parameter for all qubits. 
This is the method rot_x_layer we have written above.
"""

def initial_step(length):
    yield prepare_plus_layer(length)

def one_step(h, jr, jc, x_half_turns, h_half_turns, j_half_turns):
    length = len(h)
    yield rot_z_layer(h, h_half_turns)
    yield rot_11_layer(jr, jc, j_half_turns)
    yield rot_x_layer(length, x_half_turns)

h, jr, jc = random_instance(3)

print("Circuit:")
circuit = cirq.Circuit()  
circuit.append(initial_step(len(h)))
circuit.append(one_step(h, jr, jc, 0.1, 0.2, 0.3))
print(circuit)
print("\n")

# Simulation
simulator = cirq.Simulator()
circuit = cirq.Circuit()
circuit.append(initial_step(len(h)))
circuit.append(one_step(h, jr, jc, 0.1, 0.2, 0.3))
circuit.append(cirq.measure(*qubits, key='x'))
results = simulator.run(circuit, repetitions=100)
print("Simulation:")
print(results.histogram(key='x'))
print("\n")

# Histogram
def energy_func(length, h, jr, jc):
    def energy(measurements):
        # Reshape measurement into array that matches grid shape.
        meas_list_of_lists = [measurements[i * length:(i + 1) * length]
                              for i in range(length)]
        # Convert true/false to +1/-1.
        pm_meas = 1 - 2 * np.array(meas_list_of_lists).astype(np.int32)

        tot_energy = np.sum(pm_meas * h)
        for i, jr_row in enumerate(jr):
            for j, jr_ij in enumerate(jr_row):
                tot_energy += jr_ij * pm_meas[i, j] * pm_meas[i + 1, j]
        for i, jc_row in enumerate(jc):
            for j, jc_ij in enumerate(jc_row):
                tot_energy += jc_ij * pm_meas[i, j] * pm_meas[i, j + 1]
        return tot_energy
    return energy
print("Histogram:")
print(results.histogram(key='x', fold_func=energy_func(3, h, jr, jc)))
print("\n")

# Expectation Value
def obj_func(result):
    energy_hist = result.histogram(key='x', fold_func=energy_func(3, h, jr, jc))
    return np.sum([k * v for k,v in energy_hist.items()]) / result.repetitions
print("Expectation value:")
print(f'Value of the objective function {obj_func(results)}')
print("\n")

# Parameterize the Ansatz
circuit = cirq.Circuit()
alpha = sympy.Symbol('alpha')
beta = sympy.Symbol('beta')
gamma = sympy.Symbol('gamma')
circuit.append(initial_step(len(h)))
circuit.append(one_step(h, jr, jc, alpha, beta, gamma))
circuit.append(cirq.measure(*qubits, key='x'))
print("Parameterized Ansatz:")
print(circuit)
print("\n")

# Sweep (collection of parameter resolvers)
sweep = (cirq.Linspace(key='alpha', start=0.1, stop=0.9, length=5)

         * cirq.Linspace(key='beta', start=0.1, stop=0.9, length=5)
         * cirq.Linspace(key='gamma', start=0.1, stop=0.9, length=5))
results = simulator.run_sweep(circuit, params=sweep, repetitions=100)
print("Sweep:")
for result in results:
    print(result.params.param_dict, obj_func(result))
print("\n")

# Find the min. (grid search)
sweep_size = 10
sweep = (cirq.Linspace(key='alpha', start=0.0, stop=1.0, length=sweep_size)

         * cirq.Linspace(key='beta', start=0.0, stop=1.0, length=sweep_size)
         * cirq.Linspace(key='gamma', start=0.0, stop=1.0, length=sweep_size))
results = simulator.run_sweep(circuit, params=sweep, repetitions=100)

min = None
min_params = None
for result in results:
    value = obj_func(result)
    if min is None or value < min:
        min = value
        min_params = result.params
print("Minimum:")
print(f'Minimum objective value is {min}.')
print("\n")
