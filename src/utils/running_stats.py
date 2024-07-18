import math
import numpy as np


class RunningStats:
    def __init__(self, mode='float'):
        if mode not in ['float', 'log']:
            raise ValueError('Either compute with regular or log statistics')
        self.mode = mode
        self.clear()

    def clear(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.min = float("inf")
        self.max = float("-inf")

    def push(self, x, batch_size=None):
        if batch_size is None:
            if self.mode == 'log':
                x = np.log(x)
        if isinstance(x, np.ndarray) and batch_size is not None:
            x = x.flatten()
            n_batch = math.ceil(x.shape[0] / batch_size)
            for i in range(n_batch):
                s = i * batch_size
                e = min((i+1) * batch_size, x.shape[0])
                self.push(x[s:e], batch_size=None)
        elif isinstance(x, np.ndarray) and batch_size is None:
            x = x.flatten()
            self.n += len(x)
            self.new_m = self.old_m + (np.mean(x, dtype=np.float64) - self.old_m) / (self.n / len(x))
            self.new_s = self.old_s + np.sum((x - self.old_m) * (x - self.new_m), dtype=np.float64)

            self.old_m = self.new_m
            self.old_s = self.new_s

            self.min = min(self.min, np.min(x))
            self.max = max(self.max, np.max(x))
        else:  # regular case
            self.n += 1
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

            self.min = min(self.min, x)
            self.max = max(self.max, x)

    def mean(self):
        if self.mode == 'log':
            m = np.exp(self.new_m)
        else:
            m = self.new_m
        return m if self.n else 0.0

    def range(self):
        if self.mode == 'log':
            return np.exp(self.min), np.exp(self.max)
        else:
            return self.min, self.max

    def variance(self, ddof=1):
        if self.mode == 'log':
            s = np.exp(self.new_s)
        else:
            s = self.new_s
        return s / (self.n - ddof) if self.n > 1 else 0.0

    def standard_deviation(self, ddof=1):
        if self.mode == 'log':
            return np.exp(self.new_s / 2) / math.sqrt(self.n - ddof)
        else:
            return math.sqrt(self.variance(ddof=ddof))

    def __getitem__(self, item):
        if item == 0:
            return self.mean()
        if item == 1:
            return self.standard_deviation()
        else:
            raise ValueError(f"Can only return mean (0) and standard deviation (1) as item, requested ({item})")

    def __repr__(self):
        return f'n: {self.n}, mean: {self.mean()}, var: {self.variance()}, sd: {self.standard_deviation()}'
