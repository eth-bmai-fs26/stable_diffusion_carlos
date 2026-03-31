"""Tests for training convergence: CLIP loss decrease, UNet loss decrease."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
from src.data import ShapeDataset
from src.models import MiniCLIP, MicroUNet, NoiseScheduler
from src.train import train_clip, train_denoiser


@pytest.fixture(scope='module')
def dataset():
    torch.manual_seed(42)
    return ShapeDataset(include_size=False, include_position=False)


class TestCLIPConvergence:
    def test_clip_loss_decreases(self, dataset):
        """CLIP loss should go from ~2.2 to < 0.5 in 100 epochs."""
        torch.manual_seed(42)
        clip = MiniCLIP()
        losses = train_clip(clip, dataset, epochs=100, lr=1e-3, tau=0.07)
        assert losses[0] > 1.5, f"Initial loss {losses[0]} too low"
        assert losses[-1] < 0.5, f"Final loss {losses[-1]} should be < 0.5"

    def test_clip_loss_monotonic_trend(self, dataset):
        """Overall trend should be decreasing (allow local bumps)."""
        torch.manual_seed(42)
        clip = MiniCLIP()
        losses = train_clip(clip, dataset, epochs=100, lr=1e-3, tau=0.07)
        # Compare first 10 avg vs last 10 avg
        first_avg = sum(losses[:10]) / 10
        last_avg = sum(losses[-10:]) / 10
        assert last_avg < first_avg * 0.1, \
            f"Last 10 avg {last_avg:.4f} not much lower than first 10 avg {first_avg:.4f}"


class TestUNetConvergence:
    def test_unet_loss_decreases(self, dataset):
        """UNet loss should decrease by > 50% in 300 epochs."""
        torch.manual_seed(42)
        clip = MiniCLIP()
        train_clip(clip, dataset, epochs=100, lr=1e-3, tau=0.07)

        scheduler = NoiseScheduler(T=50)
        unet = MicroUNet()
        losses = train_denoiser(unet, dataset, scheduler,
                                clip.text_encoder, epochs=300, lr=1e-4)

        initial = sum(losses[:10]) / 10
        final = sum(losses[-10:]) / 10
        decrease_pct = (initial - final) / initial
        assert decrease_pct > 0.5, \
            f"UNet loss decrease {decrease_pct:.1%} < 50%"
