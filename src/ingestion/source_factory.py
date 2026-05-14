"""Resolve a source string to the appropriate frame-reader implementation.

Currently supported:
    - rtsp://...               → StreamReader (FFmpeg, NVDEC hint)
    - http(s)://....m3u8       → StreamReader (HLS via FFmpeg)
    - http(s)://....(mp4|webm) → StreamReader (FFmpeg)
    - http(s)://....webp       → WebPFrameReader (downloaded to temp file)
    - /path/to/clip.webp       → WebPFrameReader
    - /path/to/clip.(mp4|webm|mov|mkv|avi) → StreamReader (FFmpeg)

Future website-URL ingestion (HTML pages embedding video) will plug in here.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .errors import UnrecoverableSourceError
from .stream import StreamReader
from .webp_reader import WebPFrameReader


log = logging.getLogger(__name__)


WEBP_EXT = ".webp"
LIVE_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "udp://", "srt://")


def _suffix(source: str) -> str:
    parsed = urlparse(source)
    path = parsed.path if parsed.scheme else source
    return Path(path).suffix.lower()


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://")) or source.startswith(LIVE_SCHEMES)


def _download_to_temp(url: str) -> Path:
    import urllib.request
    suffix = _suffix(url) or ".bin"
    tmp = Path(tempfile.gettempdir()) / f"baccarat_src_{abs(hash(url))}{suffix}"
    log.info("downloading %s → %s", url, tmp)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    return tmp


def build_source(
    source: str,
    target_fps: int = 30,
    nvdec: bool = True,
    socket_timeout_s: float = 5.0,
    loop: bool = True,
    use_native_durations: bool = True,
):
    """Pick and construct the right reader for a given source string.

    Raises:
        UnrecoverableSourceError: when a local file path doesn't exist or
            points to a directory — the watchdog uses this to stop retrying.
    """
    is_url = _is_url(source)

    if not is_url:
        p = Path(source).expanduser()
        if not p.exists():
            cwd = Path.cwd()
            raise UnrecoverableSourceError(
                f"source file not found: {p}\n"
                f"  resolved to:  {p.resolve() if p.parent.exists() else p}\n"
                f"  current dir:  {cwd}\n"
                f"  hint: drop your .webp/.mp4 at the path you passed via --source, "
                f"or pass a full path."
            )
        if p.is_dir():
            try:
                listed = ", ".join(sorted(c.name for c in p.iterdir())[:8]) or "(empty)"
            except OSError:
                listed = "?"
            raise UnrecoverableSourceError(
                f"source is a directory, expected a video file: {p}\n"
                f"  directory contains: {listed}\n"
                f"  hint: point --source at the file inside, e.g. {p}/clip.webp"
            )
        source = str(p)

    ext = _suffix(source)
    is_webp = ext == WEBP_EXT
    is_live_proto = source.startswith(LIVE_SCHEMES)

    if is_webp:
        local_path = source
        if source.startswith(("http://", "https://")):
            local_path = str(_download_to_temp(source))
        return WebPFrameReader(
            path=local_path,
            target_fps=target_fps,
            loop=loop,
            use_native_durations=use_native_durations,
        )

    is_finite = (not is_live_proto) and (not source.endswith(".m3u8"))
    if is_url and source.endswith(".m3u8"):
        is_finite = False
    if not is_url:
        is_finite = True

    return StreamReader(
        source=source,
        nvdec=nvdec,
        socket_timeout_s=socket_timeout_s,
        target_fps=target_fps,
        is_finite=is_finite,
        loop=is_finite and loop,
    )


def describe_source(source: str) -> str:
    ext = _suffix(source)
    if ext == WEBP_EXT:
        return f"webp ({'remote' if _is_url(source) else 'local'})"
    if source.startswith(LIVE_SCHEMES):
        return source.split("://", 1)[0]
    if source.endswith(".m3u8"):
        return "hls"
    if _is_url(source):
        return "http"
    return "file"
