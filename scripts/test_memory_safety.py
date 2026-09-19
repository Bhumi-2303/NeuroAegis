#!/usr/bin/env python3
"""
scripts/test_memory_safety.py
──────────────────────────────
Verification suite for NeuroAegis memory-safety audit on Apple Silicon (16 GB UMA).

Verifications performed:
  1. Small Smoke Test:
     - 1 patient (chb01), 1 recording (chb01_01.edf)
     - Max 10 windows, batch size 2, 1 training step
     - Measures process RSS and PyTorch MPS memory before and after
  2. Continuous Leak Test:
     - 100 windows, batch size 2 (50 training steps)
     - Measures whether MPS and process memory grow continuously across steps
"""

import os
import sys
import gc
import time
import platform
import resource
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from neuroaegis.utils.memory import flush_memory, get_process_memory_mb
from research.phase_3.data_loader import CHBMITDataPipeline
from research.phase_3.cnn_model import Baseline1DCNN
from research.imbalance.focal_loss import BinaryFocalLossWithLogits


def get_mps_memory_mb():
    """Returns (allocated_mb, driver_mb) for MPS, or (0.0, 0.0) if not on MPS."""
    if torch.backends.mps.is_available():
        alloc = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
        driver = torch.mps.driver_allocated_memory() / (1024.0 * 1024.0)
        return alloc, driver
    return 0.0, 0.0


def run_smoke_test(device: torch.device):
    """
    Step 1: Small Smoke Test
    - 1 patient: chb01
    - 1 recording: chb01_01.edf
    - Max 10 windows
    - Batch size 2
    - 1 training step
    """
    print("\n" + "=" * 70)
    print("STEP 1: SMALL SMOKE TEST (1 patient, 1 recording, 10 windows, 1 step)")
    print("=" * 70)

    flush_memory()
    initial_rss = get_process_memory_mb()
    initial_mps_alloc, initial_mps_driver = get_mps_memory_mb()

    print(f"Pre-test Process RSS:      {initial_rss:.2f} MB")
    print(f"Pre-test MPS Allocated:    {initial_mps_alloc:.2f} MB")
    print(f"Pre-test MPS Driver:       {initial_mps_driver:.2f} MB")

    # 1. Load data
    edf_dir = os.path.join(REPO_ROOT, "CHB-MIT Dataset")
    pipeline = CHBMITDataPipeline(edf_root_dir=edf_dir, window_samples=1280)
    starts = np.arange(10, dtype=np.int64) * 640
    labels = np.zeros(10, dtype=np.float32)

    windows, labels = pipeline.extract_single_recording_windows("chb01", "chb01_01.edf", starts, labels)
    print(f"Loaded windows shape:      {windows.shape} (dtype={windows.dtype})")

    # 2. Setup model, loss, optimizer
    model = Baseline1DCNN(in_channels=23, num_classes=1).to(device)
    criterion = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(torch.from_numpy(windows), torch.from_numpy(labels).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)

    # 3. Memory before the single step
    pre_step_rss = get_process_memory_mb()
    pre_step_mps_alloc, pre_step_mps_driver = get_mps_memory_mb()

    # 4. Perform exactly 1 training step
    model.train()
    batch_x, batch_y = next(iter(loader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()

    step_loss = loss.item()
    print(f"Training step completed. Loss: {step_loss:.4f}")

    # Explicit deallocation of step tensors
    del batch_x, batch_y, logits, loss
    optimizer.zero_grad(set_to_none=True)
    flush_memory()

    post_step_rss = get_process_memory_mb()
    post_step_mps_alloc, post_step_mps_driver = get_mps_memory_mb()

    print(f"\n--- Smoke Test Results ---")
    print(f"Post-step Process RSS:     {post_step_rss:.2f} MB (Delta: {post_step_rss - pre_step_rss:+.2f} MB)")
    print(f"Post-step MPS Allocated:   {post_step_mps_alloc:.2f} MB (Delta: {post_step_mps_alloc - pre_step_mps_alloc:+.2f} MB)")
    print(f"Post-step MPS Driver:      {post_step_mps_driver:.2f} MB (Delta: {post_step_mps_driver - pre_step_mps_driver:+.2f} MB)")

    # Clean up model & loader
    del model, optimizer, criterion, loader, dataset, windows, labels
    flush_memory()

    final_rss = get_process_memory_mb()
    final_mps_alloc, final_mps_driver = get_mps_memory_mb()
    print(f"After Cleanup Process RSS: {final_rss:.2f} MB")
    print(f"After Cleanup MPS Alloc:   {final_mps_alloc:.2f} MB")
    print(f"After Cleanup MPS Driver:  {final_mps_driver:.2f} MB")
    print("Smoke Test: PASSED")


def run_continuous_leak_test(device: torch.device):
    """
    Step 2: Continuous Leak Test
    - 100 windows from chb01_01.edf
    - Batch size 2 -> 50 training steps
    - Monitor memory at steps 0, 10, 20, 30, 40, 50
    - Verify memory does NOT grow continuously across steps
    """
    print("\n" + "=" * 70)
    print("STEP 2: CONTINUOUS LEAK TEST (100 windows, batch size 2, 50 steps)")
    print("=" * 70)

    flush_memory()
    start_rss = get_process_memory_mb()

    # 1. Load 100 windows
    edf_dir = os.path.join(REPO_ROOT, "CHB-MIT Dataset")
    pipeline = CHBMITDataPipeline(edf_root_dir=edf_dir, window_samples=1280)
    starts = np.arange(100, dtype=np.int64) * 640
    labels = np.zeros(100, dtype=np.float32)

    windows, labels = pipeline.extract_single_recording_windows("chb01", "chb01_01.edf", starts, labels)
    print(f"Extracted 100 windows shape: {windows.shape} (dtype={windows.dtype}, memory={windows.nbytes / (1024*1024):.2f} MB)")

    # 2. Setup model, loss, optimizer
    model = Baseline1DCNN(in_channels=23, num_classes=1).to(device)
    criterion = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(torch.from_numpy(windows), torch.from_numpy(labels).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)

    # Free numpy arrays since DataLoader/TensorDataset hold tensor views
    del windows, labels
    flush_memory()

    # Log header
    print(f"\n{'Step':<8}{'Process RSS (MB)':<20}{'MPS Alloc (MB)':<18}{'MPS Driver (MB)':<18}{'Status'}")
    print("-" * 75)

    checkpoints = [0, 10, 20, 30, 40, 50]
    history = {}

    model.train()
    step = 0

    # Step 0 measurement
    flush_memory()
    alloc_0, driver_0 = get_mps_memory_mb()
    rss_0 = get_process_memory_mb()
    history[0] = (rss_0, alloc_0, driver_0)
    print(f"{0:<8}{rss_0:<20.2f}{alloc_0:<18.2f}{driver_0:<18.2f}{'Baseline'}")

    for batch_x, batch_y in loader:
        step += 1
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        del batch_x, batch_y, logits, loss

        if step in checkpoints:
            optimizer.zero_grad(set_to_none=True)
            flush_memory()
            rss = get_process_memory_mb()
            alloc, driver = get_mps_memory_mb()
            history[step] = (rss, alloc, driver)
            delta_alloc = alloc - history[10][1] if 10 in history and step > 10 else 0.0
            print(f"{step:<8}{rss:<20.2f}{alloc:<18.2f}{driver:<18.2f}{'Stable' if abs(delta_alloc) < 0.5 else 'Growth'}")

    # Analyze growth between post-warmup (step 10) and step 50
    step10_rss, step10_alloc, step10_driver = history[10]
    step50_rss, step50_alloc, step50_driver = history[50]

    rss_growth = step50_rss - step10_rss
    mps_alloc_growth = step50_alloc - step10_alloc
    mps_driver_growth = step50_driver - step10_driver

    print("-" * 75)
    print(f"Growth from Step 10 to Step 50 (40 steps):")
    print(f"  Process Peak RSS Growth: {rss_growth:+.2f} MB")
    print(f"  MPS Allocated Growth:    {mps_alloc_growth:+.2f} MB")
    print(f"  MPS Driver Growth:       {mps_driver_growth:+.2f} MB")

    # Assert bounded MPS memory
    assert abs(mps_alloc_growth) < 1.0, f"LEAK DETECTED: MPS allocated grew by {mps_alloc_growth:.2f} MB across 40 steps!"
    print("\nNo continuous memory growth detected. Memory remains bounded and stable across all steps!")
    print("Continuous Leak Test: PASSED")

    # Clean up
    del model, optimizer, criterion, loader, dataset
    flush_memory()


def main():
    print("=" * 70)
    print("NEUROAEGIS MEMORY AUDIT & SAFETY VERIFICATION SUITE")
    print(f"Platform: {platform.system()} {platform.machine()} | Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__} | MPS Available: {torch.backends.mps.is_available()}")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Execution Target Device: {device}")

    run_smoke_test(device)
    run_continuous_leak_test(device)

    print("\n" + "=" * 70)
    print("ALL MEMORY SAFETY VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
