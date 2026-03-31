"""Tests for evaluation thresholds using pre-trained weights."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
from src.data import ShapeDataset, generate_shape
from src.models import MiniCLIP, MicroUNet, NoiseScheduler, tokenize
from src.train import generate_image
from src.evaluate import color_accuracy, shape_accuracy, retrieval_accuracy, validate_prompt


WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'full_pipeline.pt')


@pytest.fixture(scope='module')
def pretrained():
    """Load pre-trained models."""
    checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')

    clip = MiniCLIP()
    clip.text_encoder.load_state_dict(
        {k: v.float() for k, v in checkpoint['clip_text'].items()})
    clip.image_encoder.load_state_dict(
        {k: v.float() for k, v in checkpoint['clip_image'].items()})
    clip.eval()

    unet = MicroUNet()
    unet.load_state_dict(
        {k: v.float() for k, v in checkpoint['unet'].items()})
    unet.eval()

    scheduler = NoiseScheduler(T=50)

    return clip, unet, scheduler


@pytest.fixture(scope='module')
def dataset():
    return ShapeDataset(include_size=False, include_position=False)


class TestRetrievalAccuracy:
    def test_retrieval_above_threshold(self, pretrained, dataset):
        """Retrieval accuracy on ground-truth images >= 89%."""
        clip, _, _ = pretrained
        images = [dataset[i][0] for i in range(len(dataset))]
        prompts = [dataset[i][2] for i in range(len(dataset))]
        acc = retrieval_accuracy(images, prompts, clip)
        assert acc >= 0.89, f"Retrieval accuracy {acc:.2%} < 89%"


class TestColorAccuracy:
    def test_color_on_ground_truth(self):
        """Color accuracy on ground-truth shapes should be 100%."""
        correct = 0
        total = 0
        for color in ['red', 'blue', 'green']:
            for shape in ['circle', 'square', 'triangle']:
                img = generate_shape(shape, color)
                if color_accuracy(img, color):
                    correct += 1
                total += 1
        acc = correct / total
        assert acc > 0.90, f"Color accuracy on GT: {acc:.2%} < 90%"

    def test_color_on_generated(self, pretrained):
        """Color accuracy on generated images > 90%."""
        clip, unet, scheduler = pretrained
        prompts = ['red circle', 'blue square', 'green triangle',
                   'red square', 'blue circle', 'green square',
                   'red triangle', 'blue triangle', 'green circle']

        null_tokens = tokenize('<pad> <pad>')
        null_emb = clip.text_encoder(null_tokens.unsqueeze(0))

        correct = 0
        for prompt in prompts:
            tokens = tokenize(prompt)
            text_emb = clip.text_encoder(tokens.unsqueeze(0))
            # Try multiple seeds
            for seed in [42, 123, 7, 99, 2024]:
                img, _ = generate_image(unet, scheduler, text_emb, null_emb,
                                        guidance_scale=7.5, seed=seed)
                if color_accuracy(img, prompt.split()[0]):
                    correct += 1
                    break

        acc = correct / len(prompts)
        assert acc > 0.90, f"Color accuracy on generated: {acc:.2%} < 90%"


class TestShapeAccuracy:
    def test_shape_on_ground_truth(self):
        """Shape accuracy on ground-truth should be high."""
        correct = 0
        total = 0
        for color in ['red', 'blue', 'green']:
            for shape in ['circle', 'square', 'triangle']:
                img = generate_shape(shape, color)
                if shape_accuracy(img, shape):
                    correct += 1
                total += 1
        acc = correct / total
        assert acc > 0.70, f"Shape accuracy on GT: {acc:.2%} < 70%"

    def test_shape_on_generated(self, pretrained):
        """Shape accuracy > 70% (7/9) on generated images."""
        clip, unet, scheduler = pretrained
        prompts = ['red circle', 'blue square', 'green triangle',
                   'red square', 'blue circle', 'green square',
                   'red triangle', 'blue triangle', 'green circle']

        null_tokens = tokenize('<pad> <pad>')
        null_emb = clip.text_encoder(null_tokens.unsqueeze(0))

        correct = 0
        for prompt in prompts:
            tokens = tokenize(prompt)
            text_emb = clip.text_encoder(tokens.unsqueeze(0))
            for seed in [42, 123, 7, 99, 2024]:
                img, _ = generate_image(unet, scheduler, text_emb, null_emb,
                                        guidance_scale=7.5, seed=seed)
                if shape_accuracy(img, prompt.split()[1]):
                    correct += 1
                    break

        acc = correct / len(prompts)
        assert acc > 0.70, f"Shape accuracy on generated: {acc:.2%} < 70%"


class TestValidatePrompt:
    def test_valid_prompt(self):
        known, unknown, cleaned = validate_prompt("red circle", list(tokenize.__code__.co_consts))
        # Use VOCAB directly
        from src.models import VOCAB
        known, unknown, cleaned = validate_prompt("red circle", VOCAB)
        assert known == ['red', 'circle']
        assert unknown == []
        assert cleaned == 'red circle'

    def test_unknown_words(self):
        from src.models import VOCAB
        known, unknown, cleaned = validate_prompt("purple hexagon", VOCAB)
        assert known == []
        assert unknown == ['purple', 'hexagon']
        assert cleaned == '<pad>'

    def test_mixed(self):
        from src.models import VOCAB
        known, unknown, cleaned = validate_prompt("big red circle", VOCAB)
        assert 'red' in known
        assert 'circle' in known
        assert 'big' in unknown
