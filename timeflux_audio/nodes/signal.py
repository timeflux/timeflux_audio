"""timeflux_audio.nodes.signal: generate signals"""

import time
import numpy as np
import pandas as pd
from timeflux.core.node import Node


class Sine(Node):
    """Generate a sinusoidal signal.

    Attributes:
        o (Port): Default output, provides DataFrame.

    Args:
        frequency (float): cycles per second. Default: ``1``.
        resolution (int): points per second. Default: ``200``.
        amplitude (float): signal amplitude. Default: ``1``.
        name (string): signal name. Default: ``sine``.

    Example:
        .. literalinclude:: /../examples/sine.yaml
           :language: yaml
    """

    def __init__(self, frequency=1, resolution=200, amplitude=1, name="sine"):
        self._frequency = frequency
        self._resolution = int(resolution)
        self._amplitude = amplitude
        self._name = name
        self._radian = 0
        self._now = time.time()

    def update(self):
        now = time.time()
        elapsed = now - self._now
        points = int(elapsed * self._resolution) + 1
        cycles = self._frequency * elapsed
        values = np.linspace(self._radian, np.pi * 2 * cycles + self._radian, points)
        signal = np.sin(values[:-1]) * self._amplitude
        timestamps = np.linspace(
            int(self._now * 1e6), int(now * 1e6), points, False, dtype="datetime64[us]"
        )[1:]
        self._now = now
        self._radian = values[-1]
        self.o.set(signal, timestamps, names=[self._name])
        self.o.meta = {"rate": self._resolution}


class Additive(Node):
    """Generate multiple sinusoidal signals and sum them together.

    Attributes:
        o (Port): Final signal output, provides DataFrame and meta.
        o_signals (Port): Individual signals output, provides DataFrame and meta.

    Args:
        frequencies (list): cycles per second for each wave. Default: ``[1, 2]``.
        amplitudes (list): signal amplitude. Default: ``[1, 1]``.
        resolution (int): points per second. Default: ``200``.
        name (string): signal name. Default: ``signal``.

    Example:
        .. literalinclude:: /../examples/additive.yaml
           :language: yaml
    """

    def __init__(
        self, frequencies=[1, 2], amplitudes=[1, 1], resolution=200, name="signal"
    ):
        if len(frequencies) != len(amplitudes):
            raise ValueError(
                "The frequencies and amplitudes arrays must be of equal length"
            )
        self._frequencies = frequencies
        self._amplitudes = amplitudes
        self._resolution = int(resolution)
        self._name = name
        self._meta = {"rate": self._resolution}
        self._radians = [0] * len(frequencies)
        self._now = time.time()

    def update(self):
        now = time.time()
        elapsed = now - self._now
        points = int(elapsed * self._resolution) + 1
        timestamps = np.linspace(
            int(self._now * 1e6), int(now * 1e6), points, False, dtype="datetime64[us]"
        )[1:]
        signals = np.zeros((len(timestamps), len(self._frequencies)))
        for index in range(len(self._frequencies)):
            cycles = self._frequencies[index] * elapsed
            values = np.linspace(
                self._radians[index], np.pi * 2 * cycles + self._radians[index], points
            )
            signals[:, index] = np.sin(values[:-1]) * self._amplitudes[index]
            self._radians[index] = values[-1]
        self._now = now
        final = signals.sum(axis=1) / len(self._frequencies)
        self.o.set(final, timestamps, names=[self._name])
        self.o_signals.set(signals, timestamps)
        self.o.meta = self._meta
        self.o_signals.meta = self._meta
