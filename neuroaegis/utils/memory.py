"""
NeuroAegis Memory Management and Leak Prevention Utilities
──────────────────────────────────────────────────────────
Provides unified functions to flush garbage collection, release MPS/CUDA
allocator cache, and accurately measure process resident memory on Apple Silicon
and Linux platforms.
"""

import gc
import os
import platform
import resource
import time
from typing import Optional, Dict, Any
import torch


def flush_memory():
    """
    Forces Python garbage collection and empties GPU/MPS caching allocators.
    Critical on Apple Silicon unified memory to prevent MPS allocator pool
    monopolizing system RAM.
    """
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


import ctypes


class MachTaskBasicInfo(ctypes.Structure):
    _fields_ = [
        ('virtual_size', ctypes.c_uint64),
        ('resident_size', ctypes.c_uint64),
        ('resident_size_max', ctypes.c_uint64),
        ('user_time_sec', ctypes.c_int64),
        ('user_time_usec', ctypes.c_int32),
        ('system_time_sec', ctypes.c_int64),
        ('system_time_usec', ctypes.c_int32),
        ('policy', ctypes.c_int32),
        ('suspend_count', ctypes.c_int32),
    ]


def get_peak_rss_mb() -> float:
    """
    Returns peak lifetime resident memory (ru_maxrss) in Megabytes.
    Handles Darwin/macOS (ru_maxrss in bytes) vs Linux (ru_maxrss in kilobytes).
    """
    rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rusage / (1024.0 * 1024.0)
    return rusage / 1024.0


def get_live_rss_mb() -> float:
    """
    Returns current live physical resident memory (RSS) in Megabytes.
    Uses Darwin Mach kernel task_info for high-precision live RSS on macOS,
    falling back to psutil or peak RSS.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass

    if platform.system() == "Darwin":
        try:
            libc = ctypes.CDLL(None)
            mach_task_self = libc.mach_task_self
            mach_task_self.restype = ctypes.c_uint32
            task_info = libc.task_info
            info = MachTaskBasicInfo()
            count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
            res = task_info(mach_task_self(), 20, ctypes.byref(info), ctypes.byref(count))
            if res == 0:
                return float(info.resident_size) / (1024.0 * 1024.0)
        except Exception:
            pass

    return get_peak_rss_mb()


def get_process_memory_mb() -> float:
    """
    Alias to get_peak_rss_mb() for backward compatibility.
    """
    return get_peak_rss_mb()


class MemoryTracker:
    """
    Context manager to profile memory consumption and delta for code blocks.
    """
    def __init__(self, tag: str = "Block", verbose: bool = True):
        self.tag = tag
        self.verbose = verbose
        self.start_rss: float = 0.0
        self.end_rss: float = 0.0
        self.start_time: float = 0.0
        self.duration: float = 0.0

    def __enter__(self):
        flush_memory()
        self.start_rss = get_process_memory_mb()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        flush_memory()
        self.end_rss = get_process_memory_mb()
        self.duration = time.time() - self.start_time
        delta = self.end_rss - self.start_rss
        if self.verbose:
            print(f"[MemoryTracker] {self.tag}: Start={self.start_rss:.2f}MB, End={self.end_rss:.2f}MB, Delta={delta:+.2f}MB ({self.duration:.2f}s)")
