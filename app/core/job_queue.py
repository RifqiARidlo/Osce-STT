"""Antrean job FIFO dengan SATU consumer.

Dibanding `asyncio.Lock()` (spesifikasi awal), single-consumer queue memberi jaminan yang sama
(GPU hanya dipakai satu job) plus informasi posisi antrean untuk Queue Monitor di client.
Pekerjaan berat dijalankan via `asyncio.to_thread` sehingga event loop FastAPI tetap responsif
melayani upload/polling dari laptop lain.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Callable

log = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, processor: Callable[[str], None]) -> None:
        self._processor = processor
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._pending: deque[str] = deque()
        self._current: str | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker(), name="gpu-worker")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, job_id: str) -> int:
        self._pending.append(job_id)
        await self._queue.put(job_id)
        return self.position(job_id)

    def position(self, job_id: str) -> int:
        """0 = sedang diproses, 1.. = urutan tunggu, -1 = tidak di antrean."""
        if job_id == self._current:
            return 0
        try:
            return self._pending.index(job_id) + 1
        except ValueError:
            return -1

    def snapshot(self) -> dict:
        return {"current": self._current, "pending": list(self._pending), "length": len(self._pending)}

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                self._pending.remove(job_id)
            except ValueError:
                pass
            self._current = job_id
            try:
                await asyncio.to_thread(self._processor, job_id)
            except Exception:  # processor seharusnya menangani error sendiri
                log.exception("Worker error pada job %s", job_id)
            finally:
                self._current = None
                self._queue.task_done()
