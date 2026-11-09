"""Sequential Memory Swapping: maksimal satu model besar menempati VRAM pada satu waktu.

policy="swap"    : saat model lain diminta, model aktif di-unload dulu (target GPU 4-6 GB).
policy="coexist" : model tetap dimuat setelah dipakai (GPU >= 8-12 GB) -> latensi lebih rendah.
Model yang sama dipakai berturut-turut (mis. dua job antre) tidak dimuat ulang.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)


def _release_cuda_cache() -> None:
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class GpuModelSlot:
    def __init__(self, factories: dict[str, Callable[[], Any]], policy: str = "swap") -> None:
        self._factories = factories
        self._policy = policy
        self._loaded: dict[str, Any] = {}
        self._lock = threading.RLock()
        self.load_times: dict[str, float] = {}

    def _unload(self, name: str) -> None:
        model = self._loaded.pop(name, None)
        if model is None:
            return
        close = getattr(model, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # unload tidak boleh menggagalkan pipeline
                log.warning("Gagal close model %s: %s", name, exc)
        del model
        gc.collect()
        _release_cuda_cache()
        log.info("Model '%s' di-unload dari VRAM", name)

    @contextmanager
    def use(self, name: str) -> Iterator[Any]:
        if name not in self._factories:
            raise KeyError(f"Model tidak terdaftar: {name}")
        with self._lock:
            if self._policy == "swap":
                for other in [n for n in self._loaded if n != name]:
                    self._unload(other)
            if name not in self._loaded:
                t0 = time.perf_counter()
                self._loaded[name] = self._factories[name]()
                self.load_times[name] = time.perf_counter() - t0
                log.info("Model '%s' dimuat dalam %.1fs", name, self.load_times[name])
            yield self._loaded[name]

    def unload_all(self) -> None:
        with self._lock:
            for name in list(self._loaded):
                self._unload(name)

    @property
    def loaded(self) -> list[str]:
        return list(self._loaded)
