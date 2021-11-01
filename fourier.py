import cirq

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2021)
np.set_printoptions(precision=3, suppress=True, linewidth=200)

def gen_balanced_function(N):
    half_size = N // 2
    f = np.ones(N)
    flip_loc = np.random.permutation(N)[:half_size]
    f[flip_loc] = -1
    return f


def gen_constant_function(N):
    flip = np.random.random() > 0.5
    f = np.ones(N) if flip else -1 * np.ones(N)
    return f

def randomized_alg(f, sample_size):
    N = len(f)
    sample_index = np.random.choice(N, size=sample_size)
    if len(set(f[sample_index])) == 2:
        return "accept"
    return "reject"

N = 128
K = 3
samples = 1000

res = pd.DataFrame()
for _ in range(samples):
    if np.random.rand() > 0.5:
        f = gen_balanced_function(N)
        dist = "B"
    else:
        f = gen_constant_function(N)
        dist = "C"
    decision = randomized_alg(f, K)
    res = res.append({
        "Distribution": dist,
        "Decision": decision,
        "Count": 1
    }, ignore_index=True)
confusion = res.pivot_table(index="Distribution",
                            columns="Decision",
                            values="Count",
                            aggfunc="sum")
# Translate the counts into percentage
confusion.div(confusion.sum(axis=1), axis=0).apply(lambda x: round(x, 4) * 100)

N = 128
K = 3
repetitions = 3
samples = 1000

res = pd.DataFrame()
for _ in range(samples):
    if np.random.rand() > 0.5:
        f = gen_balanced_function(N)
        dist = "B"
    else:
        f = gen_constant_function(N)
        dist = "C"
    accept_minus_reject_count = 0
    for _ in range(repetitions):
        decision = randomized_alg(f, K)
        accept_minus_reject_count += 1 if decision == "accept" else -1
    final_decision = "accept" if accept_minus_reject_count > 0 else "reject"
    res = res.append(
        {
            "Distribution": dist,
            "Decision": final_decision,
            "Count": 1
        }, ignore_index=True)
confusion = res.pivot_table(index="Distribution",
                            columns="Decision",
                            values="Count",
                            aggfunc="sum")
# Translate the counts into percentage
confusion.div(confusion.sum(axis=1), axis=0).apply(lambda x: round(x, 4) * 100)

def bitwise_dot(x, y):
    """Compute the dot product of two integers bitwise."""

    def bit_parity(i):
        n = bin(i).count("1")
        return int(n % 2)

    return bit_parity(x & y)


def fourier_transform_over_z2(v):
    """Fourier transform function over z_2^n group.

    Args:
        v: an array with 2**n elements.

    Returns:
        vs: a numpy array with same length as input.
    """
    assert len(v) & (len(v) - 1) == 0  # make sure v is 2**n long vector
    N = len(v)
    v_hat = np.array([0.0] * N)
    for y in range(N):
        for x in range(N):
            v_hat[y] += ((-1)**bitwise_dot(x, y)) * v[x]
    return v_hat / np.sqrt(N)

def bitwise_dot(x, y):
    """Compute the dot product of two integers bitwise."""

    def bit_parity(i):
        n = bin(i).count("1")
        return int(n % 2)

    return bit_parity(x & y)


def fourier_transform_over_z2(v):
    """Fourier transform function over z_2^n group.

    Args:
        v: an array with 2**n elements.

    Returns:
        vs: a numpy array with same length as input.
    """
    assert len(v) & (len(v) - 1) == 0  # make sure v is 2**n long vector
    N = len(v)
    v_hat = np.array([0.0] * N)
    for y in range(N):
        for x in range(N):
            v_hat[y] += ((-1)**bitwise_dot(x, y)) * v[x]
    return v_hat / np.sqrt(N)

f = np.array([1, -1, 1, -1])
f_hat = fourier_transform_over_z2(f)
print(f"f: {list(f)} f_hat: {list(f_hat)}")

f = np.array([1, 1, 1, -1])
f_hat = fourier_transform_over_z2(f)
print(f"f: {list(f)} f_hat: {list(f_hat)}")

f = np.array([1, -1, -1, 1])
f_hat = fourier_transform_over_z2(f)
print(f"f: {list(f)} f_hat: {list(f_hat)}")

def get_correlation(f, g):
    return f.dot(g) / np.linalg.norm(f) / np.linalg.norm(g)


def get_forrelation(f, g):
    g_hat = fourier_transform_over_z2(g)
    return f.dot(g_hat) / np.linalg.norm(f) / np.linalg.norm(g)

# let's see some examples to gain some insights of forrelation
f = np.array([1, -1, 1, -1]) 
g = np.array([1, -1, 1, -1])
print(f"Correlation: {get_correlation(f,g)}  Forrelation: {get_forrelation(f,g)}")

f = np.array([1, 1, 1, -1])
g = np.array([-1, -1, -1, 1])
print(f"Correlation: {get_correlation(f,g)}  Forrelation: {get_forrelation(f,g)}")

f = np.array([1, -1, -1, 1])
g = np.array([1, 1, 1, 1])
print(f"Correlation: {get_correlation(f,g)}  Forrelation: {get_forrelation(f,g)}")

n = 6
N = 2 ** n

# We can find a forrelated pair "as promised" through while-loop
def draw_two_distribution_from_f_set(N):
    sgn = lambda x: 1 if x >= 0 else -1
    forrelation = 0.2
    while (abs(forrelation)**2 < 0.05) and (abs(forrelation)**2 > 0.01):
        vs = np.array([np.random.normal() for _ in range(N)])
        vs_hat = fourier_transform_over_z2(vs)
        fs = np.array([sgn(v) for v in vs])
        gs = np.array([sgn(v_hat) for v_hat in vs_hat])
        forrelation = get_forrelation(fs, gs)
        correlation = get_correlation(fs, gs)
    return fs, gs, forrelation, correlation


def draw_two_distribution_from_u_set(N):
    sgn = lambda x: 1 if x >= 0 else -1
    forrelation = 0.2
    while (abs(forrelation)**2 < 0.05) and (abs(forrelation)**2 > 0.01):
        vs = np.array([np.random.normal() for _ in range(N)])
        fs = np.array([sgn(v) for v in vs])
        us = np.array([np.random.normal() for _ in range(N)])
        gs = np.array([sgn(u) for u in us])
        forrelation = get_forrelation(fs, gs)
        correlation = get_correlation(fs, gs)
    return fs, gs, forrelation, correlation

fs, gs, forrelation, correlation = draw_two_distribution_from_u_set(N)
print(f"fs: {list(fs)}")
print(f"gs: {list(gs)}")

plt.figure(figsize=(15, 5))
plt.stem(fs, use_line_collection=True)
plt.stem(gs, linefmt='--r', markerfmt='ro', use_line_collection=True)
plt.title(f"Correlation: {correlation} Forrelation: {forrelation}")

fs, gs, forrelation, correlation = draw_two_distribution_from_f_set(N)
print(f"fs: {list(fs)}")
print(f"gs: {list(gs)}")
plt.figure(figsize=(15, 5))
plt.stem(fs, use_line_collection=True)
plt.stem(gs, linefmt='--r', markerfmt="ro", use_line_collection=True)
plt.title(f"Correlation: {correlation} Forrelation: {forrelation}")

# Quantum Fourier Checking

def oracle(fs, qubits):
    return cirq.MatrixGate(np.diag(fs).astype(complex))(*qubits)


def fourier_checking_algorithm(qubits, fs, gs):
    """Returns the circuit for Fourier Checking algorithm given an input."""
    yield cirq.parallel_gate_op(cirq.H, *qubits)
    yield oracle(fs, qubits)
    yield cirq.parallel_gate_op(cirq.H, *qubits)
    yield oracle(gs, qubits)
    yield cirq.parallel_gate_op(cirq.H, *qubits)
    yield cirq.measure(*qubits)


qubits = cirq.LineQubit.range(n)
fs, gs, forrelation, correlation = draw_two_distribution_from_f_set(N)
circuit = cirq.Circuit(fourier_checking_algorithm(qubits, fs, gs))
print(circuit)

assert np.isclose(circuit.final_state_vector()[0], forrelation)

s = cirq.Simulator()
for step in s.simulate_moment_steps(circuit):
    print(step.dirac_notation())
    print("|0> state probability to observe: ",
          np.abs(step.state_vector()[0])**2)

# note that final state is not obtainable in reality

final_state = circuit.final_state_vector()
plt.figure("Probability")
plt.fill_between(np.arange(len(final_state)),
                 np.abs(final_state)**2)
plt.xlabel("State of qubits")
plt.ylabel("Probability")
plt.show()

repetitions = 100
obs = s.run(circuit, repetitions=repetitions)
qubits_name = ','.join(str(i) for i in range(n))
times_zero_was_measured = len(obs.data[obs.data[qubits_name] == 0])
print(
    f"times zero state was measured from {repetitions} measurements:" +
    f"{times_zero_was_measured} - {float(times_zero_was_measured/repetitions)*100}%"
)
if float(times_zero_was_measured / repetitions) > 0.05:
    print("fs and gs is forrelated!")

res = pd.DataFrame()
repetitions = 100
num_rounds = 1000
for _ in range(num_rounds):
    if np.random.rand() > 0.5:
        fs, gs, _, _ = draw_two_distribution_from_f_set(N)
        source = "F set"
    else:
        fs, gs, _, _ = draw_two_distribution_from_u_set(N)
        source = "U set"

    circuit = cirq.Circuit(fourier_checking_algorithm(qubits, fs, gs))
    obs = s.run(circuit, repetitions=repetitions)
    times_zero_was_measured = len(obs.data[obs.data[','.join(
        str(i) for i in range(n))] == 0])
    if times_zero_was_measured / repetitions > 0.05:
        res = res.append({
            "Source": source,
            "Decision": "accept",
            "Count": 1
        }, ignore_index=True)
    else:
        res = res.append({
            "Source": source,
            "Decision": "reject",
            "Count": 1
        }, ignore_index=True)
confusion = res.pivot_table(index="Source", columns="Decision", values="Count", aggfunc="sum")
# Translate the counts into percentage
confusion.div(confusion.sum(axis=1), axis=0).apply(lambda x: round(x, 4) * 100)

