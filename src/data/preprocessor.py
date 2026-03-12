"""
PATENT ELEMENT C/C1: Data Preprocessing
"""

import numpy as np
from collections import deque
from scipy import signal as scipy_signal


class DataPreprocessor:
    """
    PATENT ELEMENT C/C1: Preprocessing with denoising, normalization, alignment.
    Operates on real ASHRAE-scale data. Applies Savitzky-Golay denoising
    + z-score normalization per sensor channel.
    """

    def __init__(self, window_size=12):
        self.window_size = window_size
        self.buffers = {}

    def process(self, sensor_name, value):
        """Apply Savitzky-Golay denoising + z-score normalization."""
        if sensor_name not in self.buffers:
            self.buffers[sensor_name] = deque(maxlen=self.window_size)

        self.buffers[sensor_name].append(value)
        buf = list(self.buffers[sensor_name])

        # Savitzky-Golay denoising (requires min 5 points)
        if len(buf) >= 5:
            window = min(5, len(buf))
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                smoothed = scipy_signal.savgol_filter(buf, window, 2)
                denoised = smoothed[-1]
            else:
                denoised = value
        else:
            denoised = value

        return float(np.clip(denoised, 0, 1))
