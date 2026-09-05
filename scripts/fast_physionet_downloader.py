#!/usr/bin/env python3
"""
fast_physionet_downloader.py
──────────────────────────────
Parallel range-request downloader for PhysioNet CHB-MIT EDF files.
Downloads large EDF files in parallel chunks using HTTP Range requests.
"""

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PHYSIONET_URL = "https://physionet.org/files/chbmit/1.0.0"

def get_file_size(url: str) -> int:
    """Get Content-Length of a remote file."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length", 0))

def download_chunk(url: str, start_byte: int, end_byte: int, out_chunk_path: Path):
    """Download a byte range [start_byte, end_byte]."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start_byte}-{end_byte}"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(out_chunk_path, "wb") as f:
        while True:
            chunk = resp.read(1048576) # 1MB
            if not chunk:
                break
            f.write(chunk)

def download_file_parallel(url: str, out_path: Path, n_threads: int = 8):
    """Download file in n_threads parallel byte ranges."""
    if out_path.exists():
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = get_file_size(url)

    if total_bytes == 0:
        # Fallback to single thread if server doesn't report size
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(out_path, "wb") as f:
            while True:
                buf = resp.read(1048576)
                if not buf:
                    break
                f.write(buf)
        return

    chunk_size = total_bytes // n_threads
    ranges = []
    chunk_files = []

    for i in range(n_threads):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < n_threads - 1 else (total_bytes - 1)
        ranges.append((start, end))
        chunk_files.append(out_path.parent / f"{out_path.name}.part{i}")

    print(f"  ↓ {out_path.name} ({total_bytes/1e6:.1f} MB in {n_threads} parallel streams)")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(download_chunk, url, r[0], r[1], cf)
            for r, cf in zip(ranges, chunk_files)
        ]
        for f in futures:
            f.result()

    # Combine parts
    with open(out_path, "wb") as outfile:
        for cf in chunk_files:
            with open(cf, "rb") as infile:
                outfile.write(infile.read())
            cf.unlink()

    elapsed = time.time() - start_time
    speed_mbps = (total_bytes / 1e6) / elapsed if elapsed > 0 else 0
    print(f"  ✓ {out_path.name} finished in {elapsed:.1f}s ({speed_mbps:.2f} MB/s)")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        download_file_parallel(sys.argv[1], Path(sys.argv[2]))
    else:
        print("Usage: fast_physionet_downloader.py <URL> <OUTPUT_PATH>")
