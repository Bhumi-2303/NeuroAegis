# Model C Validation & Audit Report

## 1. Executive Summary

**FAIL**

The audit was halted immediately due to a critical verification failure in the frozen model checkpoint. The SHA256 hash of the provided checkpoint does not match the expected project hash.

## 2. Frozen Model Verification

**Problem**: Checkpoint hash mismatch.
**Expected Hash**: `2ec84897c39d31d68cfbf1e5c8708d5073fe19450576bf4f9132929c1832ca`
**Actual Hash**: `2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca`
**File**: `artifacts/checkpoints/frozen_cnn_gnn_gru.pt`

**Severity**: CRITICAL
**Impact**: The integrity of Model C cannot be guaranteed. The file has been modified or corrupted, rendering all subsequent evaluation, metrics, and comparisons invalid.
**Recommended action**: Locate the correct, original frozen checkpoint matching the expected SHA256 hash before proceeding with the audit.

## 18. Problems Found

- **Problem**: Checkpoint hash mismatch
- **Evidence**: `sha256sum artifacts/checkpoints/frozen_cnn_gnn_gru.pt` returned `2ec84897c39d31d68cfbf1f7e5c8708d5073fe19450576bf4f9132929c1832ca` instead of the expected `2ec84897c39d31d68cfbf1e5c8708d5073fe19450576bf4f9132929c1832ca`.
- **Severity**: Critical
- **Impact**: Audit cannot proceed because the model weights are not the expected frozen weights.
- **Recommended action**: Restore the original checkpoint and restart the audit.

## 20. Final Recommendation

**RESULTS INVALID — REBUILD REQUIRED**

*Note: As per the audit instructions ("STOP and report immediately if: checkpoint hash mismatch... Do NOT continue by making assumptions"), all other audit checks were suspended.*
