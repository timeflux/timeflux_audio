"""timeflux_audio.nodes.device: handle audio playback"""

import time
import numpy as np
import pandas as pd
import sounddevice as sd
from threading import Thread, Lock
from timeflux.core.node import Node


class Input(Node):
    """Audio input.

    Attributes:
        o (Port): Default output, provides DataFrame.

    Args:
        device (int|string): input device (numeric or string ID).
            If none specified, will use the system default. Default: ``None``.

    Example:
        .. literalinclude:: /../examples/passthrough.yaml
           :language: yaml

    """

    def __init__(self, device=None):
        self.device = device
        self.logger.info(f"Available audio interfaces:\n{sd.query_devices()}")
        self._data = np.empty((0, 1))
        self._running = True
        self._lock = Lock()
        self._thread = Thread(target=self._loop).start()

    def _callback(self, indata, frames, time, status):
        if status:
            self.logger.warning(status)
        size = indata.shape[0]
        if size > 0:
            self._lock.acquire()
            self._data = np.vstack((self._data, indata))
            self._stop = pd.Timestamp.now(tz="UTC")
            self._lock.release()

    def _loop(self):
        samplerate = sd.query_devices(self.device, "input")["default_samplerate"]
        self.meta = {"rate": samplerate}
        self._delta = 1 / samplerate
        with sd.InputStream(
            device=self.device,
            channels=1,
            callback=self._callback,
            samplerate=samplerate,
        ):
            while self._running:
                sd.sleep(1)

    def update(self):
        self._lock.acquire()
        if self._data.shape[0] > 0:
            periods = self._data.shape[0]
            start = self._stop - pd.Timedelta(self._delta * periods, "s")
            timestamps = pd.date_range(start=start, end=self._stop, periods=periods)
            self.o.set(self._data, timestamps, meta=self.meta)
            self._data = np.empty((0, 1))
        self._lock.release()

    def terminate(self):
        self._running = False


class Output(Node):
    """Audio output.

    Attributes:
        i (Port): Default input, expects DataFrame.

    Args:
        device (int|string): output device (numeric or string ID).
            If none specified, will use the system default. Default: ``None``.
        amplitude (float): audio volume.
            Default: 1

    Example:
        .. literalinclude:: /../examples/sine.yaml
           :language: yaml
    """

    def __init__(self, device=None, amplitude=1):
        self.device = device
        self.amplitude = amplitude
        self.logger.info(f"Available audio interfaces:\n{sd.query_devices()}")
        self._data = np.empty((0, 1))
        self._running = True
        self._lock = Lock()
        self._thread = Thread(target=self._loop).start()

    def _callback(self, outdata, frames, time, status):
        if status:
            self.logger.warning(status)
        size = outdata.shape[0]
        if self._data.shape[0] >= size:
            self._lock.acquire()
            outdata[:] = self._data[:size]
            self._data = self._data[size:]
            self._lock.release()
        else:
            outdata[:] = np.zeros((size, 1))

    def _loop(self):
        samplerate = sd.query_devices(self.device, "output")["default_samplerate"]
        with sd.OutputStream(
            device=self.device,
            channels=1,
            callback=self._callback,
            samplerate=samplerate,
        ):
            while self._running:
                sd.sleep(1)

    def update(self):
        if self.i.ready():
            self.i.data *= self.amplitude
            self._lock.acquire()
            self._data = np.vstack((self._data, self.i.data.values))
            self._lock.release()

    def terminate(self):
        self._running = False
