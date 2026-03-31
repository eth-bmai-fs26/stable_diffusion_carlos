"""Tests for model input/output dimensions, L2 normalization, and schedule."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
from src.models import (
    TextEncoder, ImageEncoder, MiniCLIP, NaiveMLP,
    CrossAttention, MicroUNet, NoiseScheduler, tokenize, VOCAB
)


class TestTokenize:
    def test_output_shape(self):
        tokens = tokenize("red circle")
        assert tokens.shape == (4,)

    def test_padding(self):
        tokens = tokenize("red")
        pad_idx = VOCAB.index('<pad>')
        assert tokens[1].item() == pad_idx

    def test_unknown_token(self):
        tokens = tokenize("purple circle")
        unk_idx = VOCAB.index('<unk>')
        assert tokens[0].item() == unk_idx


class TestTextEncoder:
    def test_output_shape(self):
        enc = TextEncoder()
        tokens = torch.randint(0, 12, (4, 4))
        out = enc(tokens)
        assert out.shape == (4, 32)

    def test_l2_normalized(self):
        enc = TextEncoder()
        tokens = torch.randint(0, 12, (4, 4))
        out = enc(tokens)
        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


class TestImageEncoder:
    def test_output_shape(self):
        enc = ImageEncoder()
        imgs = torch.randn(4, 3, 32, 32)
        out = enc(imgs)
        assert out.shape == (4, 32)

    def test_l2_normalized(self):
        enc = ImageEncoder()
        imgs = torch.randn(4, 3, 32, 32)
        out = enc(imgs)
        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_param_count(self):
        enc = ImageEncoder()
        params = sum(p.numel() for p in enc.parameters())
        assert 190_000 < params < 210_000, f"ImageEncoder has {params} params"


class TestMiniCLIP:
    def test_forward(self):
        clip = MiniCLIP()
        imgs = torch.randn(4, 3, 32, 32)
        tokens = torch.randint(0, 12, (4, 4))
        text_emb, img_emb = clip(imgs, tokens)
        assert text_emb.shape == (4, 32)
        assert img_emb.shape == (4, 32)

    def test_loss_computes(self):
        clip = MiniCLIP()
        text_emb = torch.randn(4, 32)
        img_emb = torch.randn(4, 32)
        loss = clip.compute_loss(text_emb, img_emb)
        assert loss.shape == ()
        assert loss.item() > 0


class TestNaiveMLP:
    def test_output_shape(self):
        mlp = NaiveMLP()
        x = torch.randn(2, 3, 32, 32)
        t_emb = torch.randn(2, 32)
        text_emb = torch.randn(2, 32)
        out = mlp(x, t_emb, text_emb)
        assert out.shape == (2, 3, 32, 32)

    def test_param_count(self):
        mlp = NaiveMLP()
        params = sum(p.numel() for p in mlp.parameters())
        assert params > 3_000_000, f"NaiveMLP should be ~3.2M params, got {params}"


class TestCrossAttention:
    def test_output_shape(self):
        ca = CrossAttention(img_dim=128, text_dim=32, head_dim=32)
        img_feat = torch.randn(2, 64, 128)
        text_feat = torch.randn(2, 3, 32)
        out, attn = ca(img_feat, text_feat)
        assert out.shape == (2, 64, 128)
        assert attn.shape == (2, 64, 3)

    def test_attention_sums_to_one(self):
        ca = CrossAttention()
        img_feat = torch.randn(2, 64, 128)
        text_feat = torch.randn(2, 3, 32)
        _, attn = ca(img_feat, text_feat)
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_has_output_projection(self):
        """CRITICAL: CrossAttention must have W_out."""
        ca = CrossAttention()
        assert hasattr(ca, 'W_out')
        assert isinstance(ca.W_out, torch.nn.Linear)


class TestMicroUNet:
    def test_output_shape(self):
        unet = MicroUNet()
        x = torch.randn(2, 3, 32, 32)
        t_emb = torch.randn(2, 32)
        text_emb = torch.randn(2, 32)
        out = unet(x, t_emb, text_emb)
        assert out.shape == (2, 3, 32, 32)

    def test_forward_with_attention(self):
        unet = MicroUNet()
        x = torch.randn(2, 3, 32, 32)
        t_emb = torch.randn(2, 32)
        text_emb = torch.randn(2, 32)
        out, attn = unet.forward_with_attention(x, t_emb, text_emb)
        assert out.shape == (2, 3, 32, 32)
        assert attn.shape[0] == 2  # batch size

    def test_param_count(self):
        unet = MicroUNet()
        params = sum(p.numel() for p in unet.parameters())
        assert 100_000 < params < 400_000, f"MicroUNet has {params} params"


class TestNoiseScheduler:
    def test_alpha_bars_decreasing(self):
        sched = NoiseScheduler(T=50)
        for i in range(1, len(sched.alpha_bars)):
            assert sched.alpha_bars[i] <= sched.alpha_bars[i-1]

    def test_alpha_bars_last_near_zero(self):
        """Cosine schedule: alpha_bar_T should be < 0.01."""
        sched = NoiseScheduler(T=50)
        assert sched.alpha_bars[-1] < 0.01, \
            f"alpha_bar_T = {sched.alpha_bars[-1]:.4f}, should be < 0.01"

    def test_betas_in_range(self):
        sched = NoiseScheduler(T=50)
        assert (sched.betas >= 0.0001).all()
        assert (sched.betas <= 0.999).all()

    def test_add_noise(self):
        sched = NoiseScheduler(T=50)
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([0, 49])
        noisy = sched.add_noise(x, t)
        assert noisy.shape == x.shape

    def test_time_embedding_shape(self):
        emb = NoiseScheduler.get_time_embedding(torch.tensor([0, 25, 49]))
        assert emb.shape == (3, 32)

    def test_T_steps(self):
        sched = NoiseScheduler(T=50)
        assert len(sched.betas) == 50
        assert len(sched.alpha_bars) == 50
