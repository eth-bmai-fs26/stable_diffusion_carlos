"""Model architectures for Stable Diffusion from Scratch course."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VOCAB = [
    'red', 'blue', 'green', 'circle', 'square', 'triangle',
    'small', 'large', 'center', 'offset', '<pad>', '<unk>'
]
WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}


def tokenize(text, max_len=4):
    """Convert text string to token indices."""
    tokens = []
    for word in text.lower().split():
        tokens.append(WORD_TO_IDX.get(word, WORD_TO_IDX['<unk>']))
    # Pad or truncate
    if len(tokens) < max_len:
        tokens += [WORD_TO_IDX['<pad>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long)


class TextEncoder(nn.Module):
    """Embedding(12,32) -> mean_pool -> Linear(32,64) -> ReLU -> Linear(64,32) -> L2_norm."""

    def __init__(self, vocab_size=12, embed_dim=32, hidden_dim=64, out_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=WORD_TO_IDX['<pad>'])
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, token_ids):
        """token_ids: (B, seq_len) -> (B, out_dim) L2-normalized."""
        x = self.embedding(token_ids)  # (B, seq_len, embed_dim)
        # Mean pool over non-padding tokens
        mask = (token_ids != WORD_TO_IDX['<pad>']).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


class ImageEncoder(nn.Module):
    """Flatten(3072) -> Linear(3072,64) -> ReLU -> Linear(64,32) -> L2_norm. ~200K params."""

    def __init__(self, input_dim=3072, out_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, images):
        """images: (B, 3, 32, 32) -> (B, out_dim) L2-normalized."""
        x = images.flatten(1)  # (B, 3072)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


class MiniCLIP(nn.Module):
    """MiniCLIP with InfoNCE loss."""

    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

    def forward(self, images, token_ids):
        text_emb = self.text_encoder(token_ids)
        image_emb = self.image_encoder(images)
        return text_emb, image_emb

    def compute_loss(self, text_emb, image_emb, tau=0.07):
        """Symmetric InfoNCE loss."""
        # Cosine similarity matrix
        logits = torch.matmul(text_emb, image_emb.t()) / tau
        labels = torch.arange(len(logits), device=logits.device)
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.t(), labels)
        return (loss_t2i + loss_i2t) / 2

    def tokenize(self, text):
        return tokenize(text)


class NaiveMLP(nn.Module):
    """Intentional failure model: no spatial inductive bias. ~3.2M params."""

    def __init__(self, text_emb_dim=32, time_emb_dim=32):
        super().__init__()
        # Input: flatten(image)=3072 + time_emb=32 + text_emb=32 = 3136
        self.fc1 = nn.Linear(3072 + time_emb_dim + text_emb_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3072)

    def forward(self, x, t_emb, text_emb):
        """
        x: (B, 3, 32, 32), t_emb: (B, 32), text_emb: (B, 32)
        Returns: (B, 3, 32, 32)
        """
        B = x.shape[0]
        flat = x.flatten(1)  # (B, 3072)
        inp = torch.cat([flat, t_emb, text_emb], dim=1)  # (B, 3136)
        h = F.relu(self.fc1(inp))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out.view(B, 3, 32, 32)


class CrossAttention(nn.Module):
    """Cross-attention: image features attend to text features.

    Q from image (dim 128), K/V from text (dim 32).
    CRITICAL: includes output projection W_out.
    """

    def __init__(self, img_dim=128, text_dim=32, head_dim=32):
        super().__init__()
        self.W_q = nn.Linear(img_dim, head_dim)
        self.W_k = nn.Linear(text_dim, head_dim)
        self.W_v = nn.Linear(text_dim, head_dim)
        self.W_out = nn.Linear(head_dim, img_dim)  # CRITICAL output projection
        self.scale = head_dim ** -0.5

    def forward(self, image_features, text_features):
        """
        image_features: (B, N_patches, img_dim)
        text_features: (B, N_tokens, text_dim)
        Returns: (output, attention_weights)
            output: (B, N_patches, img_dim)
            attention_weights: (B, N_patches, N_tokens)
        """
        Q = self.W_q(image_features)   # (B, N_patches, head_dim)
        K = self.W_k(text_features)    # (B, N_tokens, head_dim)
        V = self.W_v(text_features)    # (B, N_tokens, head_dim)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)  # (B, N_patches, N_tokens)

        out = torch.matmul(attn_weights, V)  # (B, N_patches, head_dim)
        out = self.W_out(out)  # (B, N_patches, img_dim)

        return out, attn_weights


class MicroUNet(nn.Module):
    """Minimal UNet with cross-attention at bottleneck. ~150K params."""

    def __init__(self, time_emb_dim=32, text_emb_dim=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 128),
            nn.ReLU(),
        )

        # Cross-attention at bottleneck
        self.cross_attn = CrossAttention(img_dim=128, text_dim=text_emb_dim,
                                         head_dim=32)

        # Decoder with skip connections
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1)  # 64+64=128 with skip
        self.final = nn.Conv2d(64, 3, 3, padding=1)  # 32+32=64 with skip

    def forward(self, x, t_emb, text_emb):
        """
        x: (B, 3, 32, 32), t_emb: (B, 32), text_emb: (B, 32)
        Returns: (B, 3, 32, 32)
        """
        out, _ = self.forward_with_attention(x, t_emb, text_emb)
        return out

    def forward_with_attention(self, x, t_emb, text_emb):
        """Returns (output, attention_weights)."""
        # Encoder
        e1 = F.relu(self.enc1(x))      # (B, 32, 32, 32)
        e2 = F.relu(self.enc2(e1))     # (B, 64, 16, 16)
        e3 = F.relu(self.enc3(e2))     # (B, 128, 8, 8)

        # Add time embedding (broadcast over spatial dims)
        t = self.time_mlp(t_emb)        # (B, 128)
        e3 = e3 + t[:, :, None, None]   # broadcast add

        # Cross-attention at bottleneck
        B, C, H, W = e3.shape
        # Reshape to patches: (B, H*W, C)
        patches = e3.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # Text features: (B, 1, text_dim) - single text embedding treated as 1 token
        text_feat = text_emb.unsqueeze(1)  # (B, 1, 32)

        attn_out, attn_weights = self.cross_attn(patches, text_feat)
        # Reshape back to spatial
        e3 = attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, 128, 8, 8)

        # Decoder with skip connections
        d1 = F.relu(self.dec1(e3))     # (B, 64, 16, 16)
        d1 = torch.cat([d1, e2], dim=1)  # (B, 128, 16, 16)
        d2 = F.relu(self.dec2(d1))     # (B, 32, 32, 32)
        d2 = torch.cat([d2, e1], dim=1)  # (B, 64, 32, 32)
        out = self.final(d2)            # (B, 3, 32, 32)

        return out, attn_weights


class NoiseScheduler:
    """Cosine noise schedule with T=50 steps."""

    def __init__(self, T=50, s=0.008):
        self.T = T
        self.s = s

        # Compute alpha_bar using cosine schedule
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bars = f / f[0]
        alpha_bars = alpha_bars.clamp(min=1e-5, max=1.0)

        # Compute betas from alpha_bars
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        betas = betas.clamp(min=0.0001, max=0.999)

        self.betas = betas.float()
        self.alphas = (1 - self.betas).float()
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).float()

    def add_noise(self, x_0, t, noise=None):
        """Add noise to x_0 at timestep t. q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bars[t]
        while alpha_bar_t.dim() < x_0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

    def sample_step(self, x_t, t, predicted_noise):
        """DDPM reverse step: sample x_{t-1} from x_t."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]

        # Reshape for broadcasting
        while beta_t.dim() < x_t.dim():
            beta_t = beta_t.unsqueeze(-1)
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        # Mean of p(x_{t-1} | x_t)
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )

        if isinstance(t, int) and t == 0:
            return mean
        if isinstance(t, torch.Tensor) and (t == 0).all():
            return mean

        # Add noise for t > 0
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise

    @staticmethod
    def get_time_embedding(t, dim=32):
        """Sinusoidal time embedding."""
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.float32)
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.unsqueeze(0).float()
        else:
            t = t.float()

        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
