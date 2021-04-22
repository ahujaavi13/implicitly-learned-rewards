import numpy as np

np.random.seed(0)


class Space:
    def __init__(self, low=(0,), high=(1,), dtype=np.uint8, size=-1):
        if size == -1:
            self.shape = np.shape(low)
        else:
            self.shape = (size,)
        self.low = np.array(low)
        self.high = np.array(high)
        self.dtype = dtype
        self.n = len(self.low)
