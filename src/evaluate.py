"""Evaluation metrics for Stable Diffusion from Scratch course."""

import torch
import numpy as np
from .models import tokenize


def color_accuracy(image, expected_color):
    """Check if the dominant color of non-black pixels matches expected.

    Args:
        image: tensor (3, 32, 32) or numpy (32, 32, 3) in [0, 1].
        expected_color: one of 'red', 'blue', 'green'.

    Returns: True if dominant channel matches expected color.
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()

    # Find non-black pixels (brightness > 0.1)
    brightness = image.mean(axis=2)
    mask = brightness > 0.1

    if mask.sum() == 0:
        return False

    # Average color of non-black pixels
    avg_color = image[mask].mean(axis=0)  # (3,) = [R, G, B]

    channel_map = {'red': 0, 'blue': 2, 'green': 1}
    expected_channel = channel_map[expected_color]
    dominant_channel = np.argmax(avg_color)

    return dominant_channel == expected_channel


def shape_accuracy(image, expected_shape):
    """Check if the generated shape matches expected using template correlation.

    Compares the generated image's brightness mask against ground-truth
    templates for each shape, returning True if the best-matching template
    is the expected shape.

    Args:
        image: tensor (3, 32, 32) or numpy (32, 32, 3) in [0, 1].
        expected_shape: one of 'circle', 'square', 'triangle'.

    Returns: True if best-matching template is the expected shape.
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()

    # Get brightness mask of generated image
    brightness = image.mean(axis=2)
    gen_mask = (brightness > 0.15).astype(np.float32)

    if gen_mask.sum() < 5:
        return False

    # Generate ground-truth templates for each shape
    from PIL import Image as PILImage, ImageDraw
    templates = {}
    for shape in ['circle', 'square', 'triangle']:
        tpl = PILImage.new('L', (32, 32), 0)
        draw = ImageDraw.Draw(tpl)
        r = 10
        cx, cy = 16, 16
        if shape == 'circle':
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)
        elif shape == 'square':
            draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=255)
        elif shape == 'triangle':
            draw.polygon([(cx, cy-r), (cx-r, cy+r), (cx+r, cy+r)], fill=255)
        templates[shape] = np.array(tpl, dtype=np.float32) / 255.0

    # Normalize masks for correlation
    gen_norm = gen_mask - gen_mask.mean()
    gen_std = gen_norm.std()
    if gen_std < 1e-6:
        return False

    best_shape = None
    best_corr = -2.0
    for shape, tpl_mask in templates.items():
        tpl_norm = tpl_mask - tpl_mask.mean()
        tpl_std = tpl_norm.std()
        if tpl_std < 1e-6:
            continue
        corr = (gen_norm * tpl_norm).sum() / (gen_std * tpl_std * gen_mask.size)
        if corr > best_corr:
            best_corr = corr
            best_shape = shape

    return best_shape == expected_shape


def retrieval_accuracy(generated_images, prompts, clip_model):
    """Compute retrieval accuracy: for each generated image, check if
    nearest text embedding matches the source prompt.

    Args:
        generated_images: list of image tensors (3, 32, 32).
        prompts: list of text prompts (same order as images).
        clip_model: trained MiniCLIP model.

    Returns: float accuracy in [0, 1].
    """
    clip_model.eval()
    with torch.no_grad():
        # Encode all prompts
        all_tokens = torch.stack([tokenize(p) for p in prompts])
        text_embs = clip_model.text_encoder(all_tokens)  # (N, 32)

        correct = 0
        for i, img in enumerate(generated_images):
            img_emb = clip_model.image_encoder(img.unsqueeze(0))  # (1, 32)
            # Cosine similarity to all text embeddings
            sims = torch.matmul(img_emb, text_embs.t()).squeeze(0)  # (N,)
            predicted_idx = sims.argmax().item()
            if predicted_idx == i:
                correct += 1

    return correct / len(prompts)


def validate_prompt(prompt, vocab):
    """Validate a text prompt against the vocabulary.

    Args:
        prompt: text string.
        vocab: list of valid tokens.

    Returns: (known_tokens, unknown_tokens, cleaned_prompt)
    """
    words = prompt.lower().split()
    known = [w for w in words if w in vocab]
    unknown = [w for w in words if w not in vocab]
    cleaned = ' '.join(known) if known else '<pad>'
    return known, unknown, cleaned
