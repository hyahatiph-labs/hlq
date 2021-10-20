import cirq
import matplotlib.pyplot as plt

single_qubit_heatmap = cirq.Heatmap({
    (cirq.GridQubit(0, 0),): 0.1,
    (cirq.GridQubit(1, 0),): 0.2,
    (cirq.GridQubit(2, 0),): 0.3,
    (cirq.GridQubit(0, 1),): 0.4,
})

_, ax = plt.subplots(figsize=(8, 8))
_ = single_qubit_heatmap.plot(ax);
plt.show()
