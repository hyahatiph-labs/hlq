import cirq
import cirq_google
import sympy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from cirq.contrib.svg import SVGCircuit

working_device = cirq_google.Bristlecone
print(working_device)
print("\n")

# create working subset

# Set the seed to determine the problem instance.
np.random.seed(seed=11)

# Identify working qubits from the device.
device_qubits = working_device.qubits
working_qubits = sorted(device_qubits)[:12]

# Populate a networkx graph with working_qubits as nodes.
working_graph = nx.Graph()
for qubit in working_qubits:
    working_graph.add_node(qubit)

# Pair up all neighbors with random weights in working_graph.
for qubit in working_qubits:
    for neighbor in working_device.neighbors_of(qubit):
        if neighbor in working_graph:
            # Generate a randomly weighted edge between them. Here the weighting
            # is a random 2 decimal floating point between 0 and 5.
            working_graph.add_edge(
                qubit, neighbor, weight=np.random.randint(0, 500) / 100
            )

nx.draw_circular(working_graph, node_size=1000, with_labels=True)
plt.figure(1)
plt.show()
print("\n")

# Generate QAOA circuit with p = 1

# Symbols for the rotation angles in the QAOA circuit.
alpha = sympy.Symbol('alpha')
beta = sympy.Symbol('beta')

qaoa_circuit = cirq.Circuit(
    # Prepare uniform superposition on working_qubits == working_graph.nodes
    cirq.H.on_each(working_graph.nodes()),

    # Do ZZ operations between neighbors u, v in the graph. Here, u is a qubit,
    # v is its neighboring qubit, and w is the weight between these qubits.
    (cirq.ZZ(u, v) ** (alpha * w['weight']) for (u, v, w) in working_graph.edges(data=True)),

    # Apply X operations along all nodes of the graph. Again working_graph's
    # nodes are the working_qubits. Note here we use a moment
    # which will force all of the gates into the same line.
    cirq.Moment(cirq.X(qubit) ** beta for qubit in working_graph.nodes()),

    # All relevant things can be computed in the computational basis.
    (cirq.measure(qubit) for qubit in working_graph.nodes()),
)
SVGCircuit(qaoa_circuit)

# estimate the cost

def estimate_cost(graph, samples):
    """Estimate the cost function of the QAOA on the given graph using the
    provided computational basis bitstrings."""
    cost_value = 0.0

    # Loop over edge pairs and compute contribution.
    for u, v, w in graph.edges(data=True):
        u_samples = samples[str(u)]
        v_samples = samples[str(v)]

        # Determine if it was a +1 or -1 eigenvalue.
        u_signs = (-1)**u_samples
        v_signs = (-1)**v_samples
        term_signs = u_signs * v_signs

        # Add scaled term to total cost.
        term_val = np.mean(term_signs) * w['weight']
        cost_value += term_val

    return -cost_value

alpha_value = np.pi / 4
beta_value = np.pi / 2
sim = cirq.Simulator()

sample_results = sim.sample(
    qaoa_circuit, 
    params={alpha: alpha_value, beta: beta_value}, 
    repetitions=20_000
)
print(f'Alpha = {round(alpha_value, 3)} Beta = {round(beta_value, 3)}')
print(f'Estimated cost: {estimate_cost(working_graph, sample_results)}')

# Outer loop optimization

# Set the grid size = number of points in the interval [0, 2π).
grid_size = 5

exp_values = np.empty((grid_size, grid_size))
par_values = np.empty((grid_size, grid_size, 2))

for i, alpha_value in enumerate(np.linspace(0, 2 * np.pi, grid_size)):
    for j, beta_value in enumerate(np.linspace(0, 2 * np.pi, grid_size)):
        samples = sim.sample(
            qaoa_circuit,
            params={alpha: alpha_value, beta: beta_value},
            repetitions=20000
        )
        exp_values[i][j] = estimate_cost(working_graph, samples)
        par_values[i][j] = alpha_value, beta_value

# visualize the cost as a function of alpha and beta

plt.figure(2)
plt.title('Heatmap of QAOA Cost Function Value')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.imshow(exp_values)
plt.show()

# Compare cuts

def output_cut(S_partition):
    """Plot and output the graph cut information."""

    # Generate the colors.
    coloring = []
    for node in working_graph:
        if node in S_partition:
            coloring.append('blue')
        else:
            coloring.append('red')

    # Get the weights
    edges = working_graph.edges(data=True)
    weights = [w['weight'] for (u,v, w) in edges]

    nx.draw_circular(
        working_graph,
        node_color=coloring,
        node_size=1000,
        with_labels=True,
        width=weights)
    plt.figure(3)
    plt.show()
    size = nx.cut_size(working_graph, S_partition, weight='weight')
    print(f'Cut size: {size}')

# extract the best control parameters found during sweep

best_exp_index = np.unravel_index(np.argmax(exp_values), exp_values.shape)
best_parameters = par_values[best_exp_index]
print(f'Best control parameters: {best_parameters}')

# sample some bitstrings and pick the best cut

# Number of candidate cuts to sample.
num_cuts = 100
candidate_cuts = sim.sample(
    qaoa_circuit,
    params={alpha: best_parameters[0], beta: best_parameters[1]},
    repetitions=num_cuts
)

# Variables to store best cut partitions and cut size.
best_qaoa_S_partition = set()
best_qaoa_T_partition = set()
best_qaoa_cut_size = -np.inf

# Analyze each candidate cut.
for i in range(num_cuts):
    candidate = candidate_cuts.iloc[i]
    one_qubits = set(candidate[candidate==1].index)
    S_partition = set()
    T_partition = set()
    for node in working_graph:
        if str(node) in one_qubits:
            # If a one was measured add node to S partition.
            S_partition.add(node)
        else:
            # Otherwise a zero was measured so add to T partition.
            T_partition.add(node)

    cut_size = nx.cut_size(
        working_graph, S_partition, T_partition, weight='weight')

    # If you found a better cut update best_qaoa_cut variables.
    if cut_size > best_qaoa_cut_size:
        best_qaoa_cut_size = cut_size
        best_qaoa_S_partition = S_partition
        best_qaoa_T_partition = T_partition

best_random_S_partition = set()
best_random_T_partition = set()
best_random_cut_size = -9999

# Randomly build candidate sets.
for i in range(num_cuts):
    S_partition = set()
    T_partition = set()
    for node in working_graph:
        if random.random() > 0.5:
            # If we flip heads add to S.
            S_partition.add(node)
        else:
            # Otherwise add to T.
            T_partition.add(node)

    cut_size = nx.cut_size(
        working_graph, S_partition, T_partition, weight='weight')

    # If you found a better cut update best_random_cut variables.
    if cut_size > best_random_cut_size:
        best_random_cut_size = cut_size
        best_random_S_partition = S_partition
        best_random_T_partition = T_partition

print('-----QAOA-----')
plt.figure(4)
output_cut(best_qaoa_S_partition)

print('\n\n-----RANDOM-----')
plt.figure(5)
output_cut(best_random_S_partition)

