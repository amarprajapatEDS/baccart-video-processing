"""Display backends: cv2 window, MJPEG HTTP server, annotated MP4 writer."""
from __future__ import annotations

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Tuple

import numpy as np


log = logging.getLogger(__name__)


class Display(Protocol):
    def show(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


class NullDisplay:
    def show(self, frame: np.ndarray) -> None:
        return

    def close(self) -> None:
        return


class WindowDisplay:
    def __init__(self, title: str = "Baccarat Vision AI"):
        self.title = title
        self._closed = False

    def show(self, frame: np.ndarray) -> None:
        if self._closed:
            return
        import cv2
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            raise KeyboardInterrupt("user requested quit via cv2 window")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass


class FileDisplay:
    def __init__(self, path: str, fps: int = 30, size: Optional[Tuple[int, int]] = None):
        self.path = str(path)
        self.fps = int(fps)
        self.size = size
        self._writer = None
        self._initialised_with: Optional[Tuple[int, int]] = None

    def _init_writer(self, w: int, h: int) -> None:
        import cv2
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.path, fourcc, float(self.fps), (w, h))
        if not self._writer.isOpened():
            log.warning("could not open file writer at %s", self.path)
            self._writer = None
            return
        self._initialised_with = (w, h)
        log.info("recording annotated stream → %s (%dx%d @ %d fps)", self.path, w, h, self.fps)

    def show(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._writer is None:
            self._init_writer(w, h)
            if self._writer is None:
                return
        if self._initialised_with != (w, h):
            return
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None


_INDEX_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>Baccarat Vision AI — live</title>
<style>
  body { margin:0; background:#0a0a0a; color:#ddd; font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif; }
  header { padding:12px 18px; border-bottom:1px solid #222; display:flex; justify-content:space-between; align-items:center; }
  header h1 { font-size:16px; margin:0; letter-spacing:0.04em; }
  header span { font-size:12px; color:#888; }
  main { display:flex; justify-content:center; padding:18px; }
  img { max-width:100%; height:auto; border:1px solid #222; background:#000; }
  footer { padding:10px 18px; font-size:11px; color:#666; border-top:1px solid #222; }
</style></head>
<body>
  <header><h1>Baccarat Vision AI — Live Watcher</h1><span>v5.8 spec · MJPEG stream</span></header>
  <main><img src="/stream.mjpg" alt="live"/></main>
  <footer>State transitions appear as banners on the stream. Press Ctrl-C in the terminal to stop.</footer>
</body></html>
"""


class MJPEGDisplay:
    """Serves the latest frame as MJPEG over HTTP at /stream.mjpg.

    The HTTP server runs in a daemon thread. The pipeline only encodes the
    latest frame to JPEG once per show() and stores it under a lock; each
    client streams from that shared buffer.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8089, jpeg_quality: int = 80, stream_fps: int = 25):
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)
        self.stream_fps = max(1, int(stream_fps))
        self._latest_jpeg: bytes = b""
        self._frame_version: int = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._start_server()

    def _start_server(self) -> None:
        display = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args, **kwargs):
                return

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = _INDEX_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last_version = -1
                    interval = 1.0 / display.stream_fps
                    try:
                        while not display._stop.is_set():
                            with display._lock:
                                version = display._frame_version
                                jpg = display._latest_jpeg
                            if jpg and version != last_version:
                                last_version = version
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                                self.wfile.write(jpg)
                                self.wfile.write(b"\r\n")
                                self.wfile.flush()
                            time.sleep(interval)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        return
                    except Exception as e:
                        log.debug("mjpeg client disconnected: %s", e)
                        return
                    return
                if self.path == "/healthz":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return
                self.send_response(404)
                self.end_headers()

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError as e:
            log.error("could not bind MJPEG server to %s:%d — %s", self.host, self.port, e)
            self._server = None
            return
        self._thread = threading.Thread(target=self._server.serve_forever, name="mjpeg-server", daemon=True)
        self._thread.start()
        log.info("MJPEG visualization at http://%s:%d/", self.host, self.port)

    def show(self, frame: np.ndarray) -> None:
        if self._server is None or self._stop.is_set():
            return
        import cv2
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            return
        b = buf.tobytes()
        with self._lock:
            self._latest_jpeg = b
            self._frame_version += 1

    def close(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None


class MultiDisplay:
    def __init__(self, backends: Iterable[Display]):
        self.backends: List[Display] = list(backends)

    def show(self, frame: np.ndarray) -> None:
        for b in self.backends:
            try:
                b.show(frame)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.warning("display backend %s failed: %s", type(b).__name__, e)

    def close(self) -> None:
        for b in self.backends:
            try:
                b.close()
            except Exception:
                pass


def build_display(cfg) -> Display:
    if not cfg.visualization.enabled:
        return NullDisplay()
    backends: List[Display] = []
    for name in cfg.visualization.backends:
        name = name.strip().lower()
        if name in ("none", "null", ""):
            continue
        if name == "window":
            backends.append(WindowDisplay(title=cfg.visualization.window_title))
        elif name == "web":
            backends.append(MJPEGDisplay(
                host=cfg.visualization.web_host,
                port=cfg.visualization.web_port,
                jpeg_quality=cfg.visualization.jpeg_quality,
                stream_fps=cfg.visualization.file_fps,
            ))
        elif name == "file":
            backends.append(FileDisplay(
                path=cfg.visualization.file_path,
                fps=cfg.visualization.file_fps,
            ))
        else:
            log.warning("unknown display backend: %s", name)
    if not backends:
        return NullDisplay()
    if len(backends) == 1:
        return backends[0]
    return MultiDisplay(backends)
