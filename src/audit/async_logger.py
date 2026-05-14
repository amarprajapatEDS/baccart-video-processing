"""Non-blocking async writer for JSON events and frame snapshots.

A dedicated worker thread drains a bounded queue. Drops oldest on overflow
so inference is never stalled by I/O.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


log = logging.getLogger(__name__)


class AsyncLogger:
    def __init__(self, log_dir: Path, snapshot_dir: Path, queue_size: int = 1024):
        self.log_dir = Path(log_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="async-logger", daemon=True)
        self._dropped = 0
        self._thread.start()

    def _run(self) -> None:
        events_path = self.log_dir / "events.jsonl"
        while not self._stop.is_set():
            try:
                kind, payload = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if kind == "event":
                    with events_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(payload) + "\n")
                elif kind == "snapshot":
                    path, frame = payload
                    self._write_image(path, frame)
            except Exception as e:
                log.warning("async logger write failed: %s", e)
            finally:
                self._q.task_done()

    @staticmethod
    def _write_image(path: Path, frame: np.ndarray) -> None:
        import cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)

    def write_event(self, payload: Dict[str, Any]) -> None:
        self._put(("event", payload))

    def write_snapshot(self, name: str, frame: np.ndarray) -> None:
        path = self.snapshot_dir / name
        self._put(("snapshot", (path, frame)))

    def _put(self, item: tuple) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self._dropped += 1

    @property
    def dropped(self) -> int:
        return self._dropped

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
