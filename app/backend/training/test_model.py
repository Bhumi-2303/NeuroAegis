#!/usr/bin/env python3
"""
Unit tests for the attention pooling model architecture.
Verifies correct tensor shapes, masking behavior, and entropy computation
with variable channel counts.

Usage:
    cd apps/api && python -m pytest training/test_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Allow imports from apps/api
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.model import (
    AttentionPooling,
    AttentionSeizureDetector,
    ChannelEncoder,
    SeizureClassifier,
)
from training.data import FeatureNormalizer, collate_fn


# ── Encoder Tests ────────────────────────────────────────────────────────────
class TestChannelEncoder:
    def test_output_shape(self):
        enc = ChannelEncoder(input_dim=57, embed_dim=32)
        x = torch.randn(8, 57)  # batch of 8 channels
        out = enc(x)
        assert out.shape == (8, 32), f"Expected (8, 32), got {out.shape}"

    def test_batched_channels(self):
        """Encoder applied to [B, N, 57] should produce [B, N, 32]."""
        enc = ChannelEncoder(input_dim=57, embed_dim=32)
        x = torch.randn(4, 23, 57)  # 4 samples, 23 channels each
        out = enc(x)
        assert out.shape == (4, 23, 32), f"Expected (4, 23, 32), got {out.shape}"

    def test_weight_sharing(self):
        """Same encoder applied to different channel counts gives consistent dims."""
        enc = ChannelEncoder(input_dim=57, embed_dim=32)
        out_1ch = enc(torch.randn(2, 1, 57))
        out_23ch = enc(torch.randn(2, 23, 57))
        assert out_1ch.shape == (2, 1, 32)
        assert out_23ch.shape == (2, 23, 32)


# ── Attention Pooling Tests ──────────────────────────────────────────────────
class TestAttentionPooling:
    def test_output_shape(self):
        pool = AttentionPooling(embed_dim=32)
        embeddings = torch.randn(4, 10, 32)
        mask = torch.ones(4, 10, dtype=torch.bool)
        z, alpha = pool(embeddings, mask)
        assert z.shape == (4, 32), f"Expected (4, 32), got {z.shape}"
        assert alpha.shape == (4, 10), f"Expected (4, 10), got {alpha.shape}"

    def test_attention_sums_to_one(self):
        pool = AttentionPooling(embed_dim=32)
        embeddings = torch.randn(4, 10, 32)
        mask = torch.ones(4, 10, dtype=torch.bool)
        _, alpha = pool(embeddings, mask)
        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5), \
            f"Attention weights should sum to 1.0, got {sums}"

    def test_masking_zeroes_padded(self):
        """Padded channels should get exactly 0 attention weight."""
        pool = AttentionPooling(embed_dim=32)
        embeddings = torch.randn(2, 10, 32)
        mask = torch.ones(2, 10, dtype=torch.bool)
        # Mask out last 5 channels for first sample
        mask[0, 5:] = False
        _, alpha = pool(embeddings, mask)
        # Padded positions should be zero
        assert torch.allclose(alpha[0, 5:], torch.zeros(5), atol=1e-7), \
            f"Padded positions should have zero weight, got {alpha[0, 5:]}"
        # Valid positions should still sum to 1
        assert torch.allclose(alpha[0, :5].sum(), torch.tensor(1.0), atol=1e-5)

    def test_single_channel(self):
        """Single channel should get attention weight of 1.0."""
        pool = AttentionPooling(embed_dim=32)
        embeddings = torch.randn(2, 1, 32)
        mask = torch.ones(2, 1, dtype=torch.bool)
        z, alpha = pool(embeddings, mask)
        assert torch.allclose(alpha, torch.ones(2, 1), atol=1e-5), \
            f"Single channel should get weight 1.0, got {alpha}"
        # Pooled output should equal the single embedding
        assert torch.allclose(z, embeddings.squeeze(1), atol=1e-5)

    def test_variable_channel_count_in_batch(self):
        """Different samples in a batch with different valid channel counts."""
        pool = AttentionPooling(embed_dim=32)
        embeddings = torch.randn(3, 23, 32)
        mask = torch.zeros(3, 23, dtype=torch.bool)
        mask[0, :5] = True    # Sample 0: 5 channels
        mask[1, :23] = True   # Sample 1: 23 channels
        mask[2, :1] = True    # Sample 2: 1 channel
        z, alpha = pool(embeddings, mask)
        assert z.shape == (3, 32)
        # Check each sample's attention sums to ~1
        for i, n_valid in enumerate([5, 23, 1]):
            assert torch.allclose(alpha[i, :n_valid].sum(), torch.tensor(1.0), atol=1e-5)
            assert torch.allclose(alpha[i, n_valid:], torch.zeros(23 - n_valid), atol=1e-7)


# ── SeizureClassifier Tests ─────────────────────────────────────────────────
class TestSeizureClassifier:
    def test_output_range(self):
        clf = SeizureClassifier(embed_dim=32)
        z = torch.randn(8, 32)
        out = clf(z)
        assert out.shape == (8, 1)
        assert (out >= 0).all() and (out <= 1).all(), \
            f"Output should be in [0, 1], got min={out.min()}, max={out.max()}"


# ── End-to-End Detector Tests ────────────────────────────────────────────────
class TestAttentionSeizureDetector:
    def test_full_forward(self):
        model = AttentionSeizureDetector(input_dim=57, embed_dim=32)
        features = torch.randn(4, 23, 57)
        mask = torch.ones(4, 23, dtype=torch.bool)
        pred, alpha = model(features, mask)
        assert pred.shape == (4, 1)
        assert alpha.shape == (4, 23)

    def test_variable_channels(self):
        model = AttentionSeizureDetector(input_dim=57, embed_dim=32)
        # Sample with 8 channels (padded to 23)
        features = torch.randn(2, 23, 57)
        mask = torch.zeros(2, 23, dtype=torch.bool)
        mask[0, :8] = True
        mask[1, :23] = True
        pred, alpha = model(features, mask)
        assert pred.shape == (2, 1)
        # Padded channels should have zero attention
        assert torch.allclose(alpha[0, 8:], torch.zeros(15), atol=1e-7)

    def test_entropy_loss_computation(self):
        model = AttentionSeizureDetector(input_dim=57, embed_dim=32)
        features = torch.randn(4, 10, 57)
        mask = torch.ones(4, 10, dtype=torch.bool)
        _, alpha = model(features, mask)
        entropy_loss = model.compute_entropy_loss(alpha, mask)
        assert entropy_loss.shape == (), "Entropy loss should be a scalar"
        assert entropy_loss.item() >= 0, "Entropy should be non-negative"

    def test_entropy_uniform_higher_than_concentrated(self):
        """Uniform attention should have higher entropy than concentrated."""
        model = AttentionSeizureDetector(input_dim=57, embed_dim=32)
        mask = torch.ones(1, 5, dtype=torch.bool)

        # Uniform distribution
        alpha_uniform = torch.ones(1, 5) / 5.0
        entropy_uniform = model.compute_entropy_loss(alpha_uniform, mask)

        # Concentrated distribution (one channel dominates)
        alpha_concentrated = torch.tensor([[0.96, 0.01, 0.01, 0.01, 0.01]])
        entropy_concentrated = model.compute_entropy_loss(alpha_concentrated, mask)

        assert entropy_uniform.item() > entropy_concentrated.item(), \
            f"Uniform entropy ({entropy_uniform:.4f}) should be > concentrated ({entropy_concentrated:.4f})"

    def test_gradient_flow(self):
        """Verify gradients flow through all components."""
        model = AttentionSeizureDetector(input_dim=57, embed_dim=32)
        features = torch.randn(4, 10, 57, requires_grad=True)
        mask = torch.ones(4, 10, dtype=torch.bool)
        pred, alpha = model(features, mask)
        loss = pred.sum()
        loss.backward()
        assert features.grad is not None, "Gradients should flow to input"
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_single_channel_forward(self):
        """Model works with just 1 channel (simulating Bonn dataset)."""
        model = AttentionSeizureDetector(input_dim=57, embed_dim=32)
        features = torch.randn(4, 1, 57)
        mask = torch.ones(4, 1, dtype=torch.bool)
        pred, alpha = model(features, mask)
        assert pred.shape == (4, 1)
        assert torch.allclose(alpha, torch.ones(4, 1), atol=1e-5)


# ── FeatureNormalizer Tests ──────────────────────────────────────────────────
class TestFeatureNormalizer:
    def test_normalize(self):
        norm = FeatureNormalizer()
        data = np.random.randn(100, 57) * 10 + 5
        normalized = norm.fit_transform(data)
        assert np.allclose(normalized.mean(axis=0), 0, atol=0.1)
        assert np.allclose(normalized.std(axis=0), 1, atol=0.1)

    def test_transform_without_fit_raises(self):
        norm = FeatureNormalizer()
        try:
            norm.transform(np.random.randn(10, 57))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ── Collate Function Tests ───────────────────────────────────────────────────
class TestCollateFn:
    def test_padding(self):
        batch = [
            {"features": torch.randn(5, 57), "label": 0, "patient_id": "p1",
             "mask": torch.ones(5, dtype=torch.bool), "n_channels": 5},
            {"features": torch.randn(10, 57), "label": 1, "patient_id": "p2",
             "mask": torch.ones(10, dtype=torch.bool), "n_channels": 10},
        ]
        out = collate_fn(batch)
        assert out["features"].shape == (2, 10, 57), f"Expected (2, 10, 57), got {out['features'].shape}"
        assert out["mask"].shape == (2, 10)
        assert out["mask"][0, :5].all()
        assert not out["mask"][0, 5:].any()
        assert out["mask"][1, :10].all()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
