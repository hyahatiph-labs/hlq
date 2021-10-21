import cirq
import matplotlib.pyplot as plt
from custom.hlgate import HLGate

hlg = HLGate()
q = cirq.LineQubit.range(4)
circuit = cirq.Circuit([hlg.on_each(*q), cirq.measure(*q)])
result = cirq.Simulator().run(circuit, repetitions=100)
_ = cirq.vis.plot_state_histogram(result, plt.subplot())

plt.show()
