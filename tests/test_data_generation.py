"""Tests for data generation: shapes, colors, values, class counts."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import torch
from src.data import generate_shape, generate_dataset, ShapeDataset, SHAPES, COLORS_MAP


class TestGenerateShape:
    """Test the generate_shape function."""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("color", list(COLORS_MAP.keys()))
    def test_shape_color_combinations(self, shape, color):
        """All 9 shape/color combos produce valid images."""
        img = generate_shape(shape, color)
        assert img.shape == (32, 32, 3)
        assert img.dtype == np.float32

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("color", list(COLORS_MAP.keys()))
    def test_values_in_range(self, shape, color):
        """All pixel values in [0, 1]."""
        img = generate_shape(shape, color)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_non_black_pixels_exist(self):
        """Generated shapes have non-black pixels."""
        for shape in SHAPES:
            for color in COLORS_MAP:
                img = generate_shape(shape, color)
                assert img.max() > 0, f"{color} {shape} is all black"

    def test_correct_color_channel(self):
        """Dominant channel matches expected color."""
        channel_map = {'red': 0, 'blue': 2, 'green': 1}
        for color, expected_ch in channel_map.items():
            img = generate_shape('circle', color)
            mask = img.mean(axis=2) > 0.1
            if mask.sum() > 0:
                avg = img[mask].mean(axis=0)
                assert np.argmax(avg) == expected_ch, \
                    f"{color}: dominant channel {np.argmax(avg)} != {expected_ch}"

    def test_sizes(self):
        """Small shapes have fewer non-black pixels than large shapes."""
        for shape in SHAPES:
            small = generate_shape(shape, 'red', size='small')
            large = generate_shape(shape, 'red', size='large')
            small_count = (small.mean(axis=2) > 0.1).sum()
            large_count = (large.mean(axis=2) > 0.1).sum()
            assert large_count > small_count, \
                f"{shape}: large ({large_count}) should have more pixels than small ({small_count})"


class TestGenerateDataset:
    """Test the generate_dataset function."""

    def test_base_class_count(self):
        """Base dataset has 9 classes."""
        dataset = generate_dataset()
        assert len(dataset) == 9

    def test_with_size_class_count(self):
        """Dataset with size has 18 classes."""
        dataset = generate_dataset(include_size=True)
        assert len(dataset) == 18

    def test_with_size_and_position_class_count(self):
        """Dataset with size+position has 36 classes."""
        dataset = generate_dataset(include_size=True, include_position=True)
        assert len(dataset) == 36

    def test_base_labels_format(self):
        """Base labels are '{color} {shape}'."""
        dataset = generate_dataset()
        for img, label in dataset:
            parts = label.split()
            assert len(parts) == 2
            assert parts[0] in COLORS_MAP
            assert parts[1] in SHAPES

    def test_extended_labels_format(self):
        """Extended labels are '{size} {color} {shape}'."""
        dataset = generate_dataset(include_size=True)
        for img, label in dataset:
            parts = label.split()
            assert len(parts) == 3
            assert parts[0] in ['small', 'large']
            assert parts[1] in COLORS_MAP
            assert parts[2] in SHAPES


class TestShapeDataset:
    """Test the ShapeDataset class."""

    def test_length(self):
        ds = ShapeDataset()
        assert len(ds) == 9

    def test_getitem_returns_tuple(self):
        ds = ShapeDataset()
        item = ds[0]
        assert len(item) == 3  # (image, label_idx, label_text)

    def test_image_tensor_shape(self):
        ds = ShapeDataset()
        img, _, _ = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)

    def test_image_values_in_range(self):
        ds = ShapeDataset()
        for i in range(len(ds)):
            img, _, _ = ds[i]
            assert img.min() >= 0.0
            assert img.max() <= 1.0

    def test_label_index_valid(self):
        ds = ShapeDataset()
        for i in range(len(ds)):
            _, idx, _ = ds[i]
            assert 0 <= idx < len(ds)

    def test_unique_labels(self):
        ds = ShapeDataset()
        labels = set()
        for i in range(len(ds)):
            _, _, text = ds[i]
            labels.add(text)
        assert len(labels) == 9
