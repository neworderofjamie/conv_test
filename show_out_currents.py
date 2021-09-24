import numpy as np
import matplotlib.pyplot as plt
import sys

out_currents = np.fromfile("outCurrents%s.bin" % sys.argv[1], dtype=np.float32)
print(np.amin(out_currents), np.amax(out_currents))
out_currents = out_currents.reshape((62, 62))
plt.imshow(out_currents)

plt.show()
