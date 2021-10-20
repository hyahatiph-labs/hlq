import cirq
from custom.hlgate import HLGate

# print the Hyahatiph Labs custom gate

"""Use the custom gate in a circuit."""
hlg = HLGate()
circ = cirq.Circuit(
    hlg.on(cirq.LineQubit(0))
)

print("Hyahatiph Labs Gate:")
print(circ)

