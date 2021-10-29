import numpy as np
import matplotlib.pyplot as plt

out_currents_procedural = np.fromfile("outCurrentsProcedural.bin", dtype=np.float32)
out_currents_toeplitz = np.fromfile("outCurrentsToeplitz.bin", dtype=np.float32)


size = np.ceil(np.sqrt(out_currents_procedural.shape[0])).astype(int)

out_currents_procedural = out_currents_procedural.reshape((size, size))
out_currents_toeplitz = out_currents_toeplitz.reshape((size, size))

fig, axes = plt.subplots(2)
axes[0].set_title("Procedural")
axes[0].imshow(out_currents_procedural)

axes[1].set_title("Toeplitz")
axes[1].imshow(out_currents_toeplitz)

fig.tight_layout(pad=0)
plt.show()
