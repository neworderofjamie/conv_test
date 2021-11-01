import numpy as np
import sys
import matplotlib.pyplot as plt

# Read number of channels from command line
num_channels = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# Load data
out_currents_procedural = np.fromfile("outCurrentsProcedural.bin", dtype=np.float32)
out_currents_toeplitz = np.fromfile("outCurrentsToeplitz.bin", dtype=np.float32)

# Check shapes match
assert out_currents_procedural.shape == out_currents_toeplitz.shape

# Reshape to get number of channels
out_currents_procedural = np.reshape(out_currents_procedural, (-1, num_channels))
out_currents_toeplitz = np.reshape(out_currents_toeplitz, (-1, num_channels))

size = np.ceil(np.sqrt(out_currents_procedural.shape[0])).astype(int)

out_currents_procedural = out_currents_procedural.reshape((size, size, num_channels))
out_currents_toeplitz = out_currents_toeplitz.reshape((size, size, num_channels))

fig, axes = plt.subplots(num_channels, 2, sharex="col", sharey="row")
for c in range(num_channels):
    axes[c, 0].set_title("Procedural")
    axes[c, 0].imshow(out_currents_procedural[:,:,c])

    axes[c, 1].set_title("Toeplitz")
    axes[c, 1].imshow(out_currents_toeplitz[:,:,c])

fig.tight_layout(pad=0)
plt.show()
