import numpy as np
import matplotlib.pyplot as plt
out_currents = np.fromfile("outCurrents.bin", dtype=np.float32)
out_currents.shape
out_currents = out_currents.reshape((62, 62))
plt.imshow(out_currents)
plt.show()
