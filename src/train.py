"""Training loops and generation pipeline for Stable Diffusion from Scratch."""

import copy
import torch
import torch.nn.functional as F
from .models import NoiseScheduler, tokenize


def train_clip(model, dataset, epochs=100, lr=1e-3, tau=0.07):
    """Train MiniCLIP with InfoNCE loss.

    Full-batch training (9 pairs for base dataset).
    Returns list of per-epoch losses.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    # Prepare full batch
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    texts = [dataset[i][2] for i in range(len(dataset))]
    token_ids = torch.stack([tokenize(t) for t in texts])

    for epoch in range(epochs):
        optimizer.zero_grad()
        text_emb, image_emb = model(images, token_ids)
        loss = model.compute_loss(text_emb, image_emb, tau=tau)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f'CLIP Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    return losses


def train_denoiser(model, dataset, scheduler, text_encoder, epochs=300,
                   lr=1e-4, is_mlp=False, steps_per_epoch=1, ema_decay=0.0):
    """Train a denoiser (MicroUNet or NaiveMLP).

    Random timestep sampling, MSE loss on noise prediction.
    Multiple gradient steps per epoch expose the model to diverse noise samples.
    When ema_decay > 0, maintains exponential moving average of weights and
    copies them back to the model at the end for better generation quality.
    Returns list of per-epoch losses.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    # EMA: maintain a smoothed copy of weights
    if ema_decay > 0:
        ema_state = copy.deepcopy(model.state_dict())

    # Prepare data
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    texts = [dataset[i][2] for i in range(len(dataset))]
    token_ids = torch.stack([tokenize(t) for t in texts])

    # Pre-compute text embeddings
    with torch.no_grad():
        text_embs = text_encoder(token_ids)  # (N, 32)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()

            # Random timesteps for each sample
            B = len(images)
            t = torch.randint(0, scheduler.T, (B,))
            noise = torch.randn_like(images)
            x_t = scheduler.add_noise(images, t, noise)

            # Time embeddings
            t_emb = scheduler.get_time_embedding(t)  # (B, 32)

            # Predict noise
            predicted_noise = model(x_t, t_emb, text_embs)

            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            # Update EMA weights
            if ema_decay > 0:
                with torch.no_grad():
                    for key, param in model.state_dict().items():
                        ema_state[key].mul_(ema_decay).add_(param, alpha=1 - ema_decay)

            epoch_loss += loss.item()

        losses.append(epoch_loss / steps_per_epoch)

        if (epoch + 1) % 50 == 0:
            print(f'Denoiser Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}')

    # Copy EMA weights back to model for generation
    if ema_decay > 0:
        model.load_state_dict(ema_state)

    return losses


@torch.no_grad()
def generate_image(model, scheduler, text_emb, null_text_emb=None,
                   guidance_scale=7.5, steps=50, seed=42):
    """Generate an image using the full denoising loop with optional CFG.

    Args:
        model: trained denoiser (MicroUNet).
        scheduler: NoiseScheduler instance.
        text_emb: (1, 32) text embedding for the prompt.
        null_text_emb: (1, 32) embedding for unconditional (empty) text.
            If None, no classifier-free guidance is applied.
        guidance_scale: CFG scale (default 7.5).
        steps: number of denoising steps.
        seed: random seed.

    Returns: (final_image, intermediates) where intermediates is a list of
        images at selected steps for filmstrip display.
    """
    torch.manual_seed(seed)
    x = torch.randn(1, 3, 32, 32)

    intermediates = []
    # Save ~8 frames for filmstrip
    save_steps = set([steps - 1] + [int(i * (steps - 1) / 7) for i in range(8)])

    for t_idx in reversed(range(steps)):
        t = torch.tensor([t_idx])
        t_emb = scheduler.get_time_embedding(t)

        if null_text_emb is not None and guidance_scale != 1.0:
            # Classifier-free guidance: two forward passes
            noise_cond = model(x, t_emb, text_emb)
            noise_uncond = model(x, t_emb, null_text_emb)
            predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            predicted_noise = model(x, t_emb, text_emb)

        x = scheduler.sample_step(x, t_idx, predicted_noise)

        if t_idx in save_steps:
            intermediates.append(x[0].clone().clamp(0, 1))

    intermediates.reverse()
    final = x[0].clamp(0, 1)
    return final, intermediates


def evaluate_generation(model, scheduler, text_encoder, prompts,
                        guidance_scale=7.5, seed=42):
    """Generate images for a list of prompts and return them.

    Args:
        model: trained MicroUNet.
        scheduler: NoiseScheduler.
        text_encoder: trained TextEncoder.
        prompts: list of text strings.
        guidance_scale: CFG scale.
        seed: random seed.

    Returns: list of (prompt, image_tensor) tuples.
    """
    # Null text embedding for CFG
    null_tokens = tokenize('<pad> <pad>')
    null_text_emb = text_encoder(null_tokens.unsqueeze(0))

    results = []
    for prompt in prompts:
        tokens = tokenize(prompt)
        text_emb = text_encoder(tokens.unsqueeze(0))
        img, _ = generate_image(model, scheduler, text_emb, null_text_emb,
                                guidance_scale=guidance_scale, seed=seed)
        results.append((prompt, img))

    return results
