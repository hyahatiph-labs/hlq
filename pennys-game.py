import cirq

"""
any time someone flips a coin they are, in some capacity, 
performing a Schrödinger's cat experiment where the coin 
can be considered simultaneously both heads and tails
"""

# Get qubits and circuit
qreg = [cirq.LineQubit(x) for x in range(2)]
circ = cirq.Circuit()

# add Hadamard and measure

circ.append(cirq.H(qreg[0]))
circ.append(cirq.H(qreg[0]))
circ.append(cirq.H(qreg[0]))
circ.append(cirq.measure(*qreg, key="f"))

# print the circuit
print("Circuit")
print(circ)


# Simulate the circuit
sim = cirq.Simulator()
print(sim.sample(circ, repetitions=30))
