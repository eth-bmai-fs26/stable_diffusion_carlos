"""Shape dataset generation for Stable Diffusion from Scratch course."""

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np

SHAPES = ['circle', 'square', 'triangle']
COLORS_MAP = {
    'red': (230, 25, 25),
    'blue': (25, 25, 230),
    'green': (25, 204, 25),
}
SIZES = {'small': 5, 'large': 10}
POSITIONS = {
    'center': (16, 16),
    'offset': [(8, 8), (8, 24), (24, 8), (24, 24)],
}


def generate_shape(shape, color, size='large', position='center', img_size=32):
    """Generate a 32x32x3 image with a colored shape.

    Returns a numpy array with values in [0, 1], shape (img_size, img_size, 3).
    """
    img = Image.new('RGB', (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    rgb = COLORS_MAP[color]
    radius = SIZES[size]

    if position == 'center':
        cx, cy = 16, 16
    else:
        rng = np.random.RandomState()
        cx, cy = POSITIONS['offset'][rng.randint(len(POSITIONS['offset']))]

    if shape == 'circle':
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(bbox, fill=rgb)
    elif shape == 'square':
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.rectangle(bbox, fill=rgb)
    elif shape == 'triangle':
        pts = [
            (cx, cy - radius),
            (cx - radius, cy + radius),
            (cx + radius, cy + radius),
        ]
        draw.polygon(pts, fill=rgb)

    return np.array(img, dtype=np.float32) / 255.0


def generate_dataset(include_size=False, include_position=False):
    """Generate the shape dataset.

    Base: 9 classes (3 shapes x 3 colors).
    With size: 18 classes.
    With size + position: 36 classes.

    Returns list of (image_array, label_text) tuples.
    """
    sizes = list(SIZES.keys()) if include_size else ['large']
    positions = ['center', 'offset'] if include_position else ['center']
    dataset = []

    for color in COLORS_MAP:
        for shape in SHAPES:
            for sz in sizes:
                for pos in positions:
                    img = generate_shape(shape, color, size=sz, position=pos)
                    if include_size:
                        label = f"{sz} {color} {shape}"
                    else:
                        label = f"{color} {shape}"
                    dataset.append((img, label))

    return dataset


class ShapeDataset(Dataset):
    """PyTorch Dataset for generated shapes."""

    def __init__(self, include_size=False, include_position=False):
        raw = generate_dataset(include_size=include_size,
                               include_position=include_position)
        self.images = []
        self.labels = []
        self.label_texts = []

        unique_labels = []
        for img, label in raw:
            if label not in unique_labels:
                unique_labels.append(label)

        self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}

        for img, label in raw:
            # Convert HWC -> CHW tensor
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            self.images.append(tensor)
            self.labels.append(self.label_to_idx[label])
            self.label_texts.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.label_texts[idx]
