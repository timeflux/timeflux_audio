"""timeflux_audio.nodes.device: handle audio playback"""

import time
import numpy as np
import sounddevice as sd
from threading import Thread, Lock
from timeflux.core.node import Node


class Output(Node):
    """Audio output.

    Attributes:
        i (Port): Default input, expects DataFrame.

    Args:
        device (int|string): output device (numeric or string ID).
            If none specified, will use the system default. Default: ``None``.

    Example:
        .. literalinclude:: /../examples/sine.yaml
           :language: yaml
    """

    def __init__(self, device=None):
        self.device = device
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
        self._start_idx = 0
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
            self._lock.acquire()
            self._data = np.vstack((self._data, self.i.data.values))
            self._lock.release()

    def terminate(self):
        self._running = False
