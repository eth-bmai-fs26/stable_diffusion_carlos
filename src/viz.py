"""Visualization functions for Stable Diffusion from Scratch course.

All functions use Okabe-Ito palette, never use color as sole encoding,
min 12pt text, include alt-text, return fig objects.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# Okabe-Ito palette
COLORS = {
    'text': '#E69F00',
    'image': '#0072B2',
    'connection': '#56B4E9',
    'failure': '#D55E00',
    'crossmodal': '#CC79A7',
    'neutral': '#999999',
}

# Standard rcParams
def _apply_style():
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.facecolor'] = 'white'


def plot_image_grid(images, titles=None, ncols=3, figsize=None, suptitle=None):
    """Display a grid of images with optional titles.

    Args:
        images: list of numpy arrays (H,W,3) or tensors (3,H,W) in [0,1].
        titles: optional list of strings.
        ncols: number of columns.
        figsize: optional figure size.
        suptitle: optional super title.

    Returns: fig
    """
    _apply_style()
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 2.5, nrows * 2.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and img.shape[0] in (1, 3):
                    img = img.permute(1, 2, 0)
                img = img.detach().cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=12)
        ax.axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_similarity_matrix(sim_matrix, row_labels, col_labels,
                           title='Cosine Similarity', cmap='Blues'):
    """Plot a similarity matrix heatmap with numeric annotations.

    Uses sequential single-hue colormap. Numbers displayed on each cell.

    Returns: fig
    """
    _apply_style()
    n_rows, n_cols = sim_matrix.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.8), max(5, n_rows * 0.8)))

    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.detach().cpu().numpy()

    im = ax.imshow(sim_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # Annotate cells with numeric values
    for i in range(n_rows):
        for j in range(n_cols):
            val = sim_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_attention_heatmap(attn_weights, text_tokens, grid_size=8,
                           title='Attention Weights'):
    """Plot attention heatmap over spatial grid for each text token.

    Args:
        attn_weights: (N_patches, N_tokens) or (B, N_patches, N_tokens)
        text_tokens: list of token strings
        grid_size: spatial grid dimension (patches = grid_size^2)

    Returns: fig
    """
    _apply_style()
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    if attn_weights.ndim == 3:
        attn_weights = attn_weights[0]  # Take first batch element

    n_tokens = len(text_tokens)
    fig, axes = plt.subplots(1, n_tokens, figsize=(3 * n_tokens, 3))
    if n_tokens == 1:
        axes = [axes]

    for i, (ax, token) in enumerate(zip(axes, text_tokens)):
        heatmap = attn_weights[:, i].reshape(grid_size, grid_size)
        im = ax.imshow(heatmap, cmap='Blues', vmin=0)
        ax.set_title(f'"{token}"', fontsize=12)
        ax.axis('off')
        # Add numeric labels for accessibility
        for r in range(grid_size):
            for c in range(grid_size):
                val = heatmap[r, c]
                if val > 0.1:
                    ax.text(c, r, f'{val:.1f}', ha='center', va='center',
                            fontsize=7, color='white' if val > 0.5 else 'black')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_denoising_filmstrip(images, steps=None, title='Denoising Process'):
    """Plot a filmstrip of denoising steps (8 frames).

    Args:
        images: list of image tensors/arrays at different timesteps.
        steps: optional list of step numbers.

    Returns: fig
    """
    _apply_style()
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2.5))
    if n == 1:
        axes = [axes]

    for i, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img = img.detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        label = f't={steps[i]}' if steps else f'Step {i}'
        ax.set_title(label, fontsize=10)
        ax.axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_training_loss(losses, title='Training Loss', xlabel='Epoch',
                       ylabel='Loss', color=None):
    """Plot training loss curve.

    Returns: fig
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    c = color or COLORS['image']
    ax.plot(losses, color=c, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_embedding_space_2d(embeddings, labels, title='Embedding Space',
                            method='pca'):
    """Plot 2D projection of embedding space.

    Args:
        embeddings: (N, D) tensor or array.
        labels: list of string labels.
        method: 'pca' projection.

    Returns: fig
    """
    _apply_style()
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Simple PCA via SVD
    centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by unique label
    unique_labels = list(dict.fromkeys(labels))
    palette = ['#E69F00', '#0072B2', '#56B4E9', '#D55E00', '#CC79A7',
               '#009E73', '#F0E442', '#999999', '#000000']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h']

    for i, ulabel in enumerate(unique_labels):
        mask = [l == ulabel for l in labels]
        idx = [j for j, m in enumerate(mask) if m]
        c = palette[i % len(palette)]
        m = markers[i % len(markers)]
        ax.scatter(proj[idx, 0], proj[idx, 1], c=c, marker=m, s=80,
                   label=ulabel, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('PC 1', fontsize=12)
    ax.set_ylabel('PC 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_report_card(metrics, title='Model Report Card'):
    """Plot a report card with bar chart of metrics.

    Args:
        metrics: dict of {name: (value, threshold)} pairs.

    Returns: fig
    """
    _apply_style()
    names = list(metrics.keys())
    values = [metrics[n][0] for n in names]
    thresholds = [metrics[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(names))
    bars = ax.bar(x, values, color=COLORS['image'], alpha=0.8, label='Achieved')

    # Add threshold lines
    for i, thresh in enumerate(thresholds):
        ax.plot([i - 0.4, i + 0.4], [thresh, thresh], color=COLORS['failure'],
                linewidth=2, linestyle='--')

    # Add numeric labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)

    # Legend
    threshold_patch = mpatches.Patch(color=COLORS['failure'], label='Threshold')
    ax.legend(handles=[bars.patches[0], threshold_patch],
              labels=['Achieved', 'Threshold'], fontsize=10)
    fig.tight_layout()
    return fig


def plot_guidance_comparison(images, scales, title='Guidance Scale Comparison'):
    """Plot images generated at different guidance scales side by side.

    Args:
        images: list of image tensors/arrays.
        scales: list of guidance scale values.

    Returns: fig
    """
    _apply_style()
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 3))
    if n == 1:
        axes = [axes]

    for ax, img, scale in zip(axes, images, scales):
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img = img.detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f'Scale={scale}', fontsize=11)
        ax.axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_progress_comparison(before_images, after_images, labels,
                             title='Before vs After'):
    """Plot before/after comparison of generated images.

    Returns: fig
    """
    _apply_style()
    n = len(labels)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5.5))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        for row, (imgs, row_label) in enumerate(
                [(before_images, 'Before'), (after_images, 'After')]):
            img = imgs[i]
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and img.shape[0] in (1, 3):
                    img = img.permute(1, 2, 0)
                img = img.detach().cpu().numpy()
            img = np.clip(img, 0, 1)
            axes[row, i].imshow(img)
            if row == 0:
                axes[row, i].set_title(labels[i], fontsize=11)
            axes[row, i].set_ylabel(row_label if i == 0 else '', fontsize=11)
            axes[row, i].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def make_filmstrip_gif(images, filepath, duration=200):
    """Save a list of images as an animated GIF.

    Args:
        images: list of numpy arrays (H,W,3) in [0,1] or tensors.
        filepath: output path for the GIF.
        duration: frame duration in ms.
    """
    from PIL import Image as PILImage

    frames = []
    for img in images:
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img = img.detach().cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        frames.append(PILImage.fromarray(img))

    if frames:
        frames[0].save(filepath, save_all=True, append_images=frames[1:],
                       duration=duration, loop=0)
