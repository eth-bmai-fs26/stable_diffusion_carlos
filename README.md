# Stable Diffusion from Scratch: A Decision-Maker's Guide

Build a working text-to-image model in 5 interactive notebooks. No AI experience required.

## Who This Is For

Managers and non-technical leaders who need to understand generative AI image systems for strategic decision-making. After this course, you will understand the CLIP, diffusion, and cross-attention architecture well enough to evaluate vendors, diagnose model failures, and ask your engineering team the right questions.

## What You'll Learn

- **Notebook 1 — The Raw Ingredients:** How computers represent images (tensors) and text (embeddings)
- **Notebook 2 — The Translator (CLIP):** How contrastive learning creates a shared language between text and images
- **Notebook 3 — The Engine (Diffusion):** How models generate images by learning to remove noise step by step
- **Notebook 4 — The Conversation (Cross-Attention):** How text steers image generation at every denoising step
- **Notebook 5 — The Road Trip (Full Pipeline):** How all components wire together, classifier-free guidance, and evaluation

## How to Use

Open each notebook in Google Colab and click "Run All." Each notebook is fully self-contained — no installation, no setup, no dependencies to manage.

| Notebook | Title | Duration |
|----------|-------|----------|
| 01 | The Raw Ingredients — Images and Text as Numbers | ~25 min |
| 02 | The Translator — CLIP | ~30 min |
| 03 | The Engine — Diffusion | ~35 min |
| 04 | The Conversation — Cross-Attention | ~30 min |
| 05 | The Road Trip — The Full Pipeline | ~30 min |

**Total time: ~2.5 hours.** Fits a half-day workshop with breaks.

## Prerequisites

None. Ability to click "Run" in Google Colab. No Python experience required.

## What You'll Build

A toy text-to-image model with ~150,000 parameters that generates 32x32 colored shapes from 2-word text prompts. It uses the **exact same architecture** as Stable Diffusion (CLIP text encoder, U-Net denoiser with cross-attention, cosine noise schedule, classifier-free guidance) — just 25,000x smaller.

## Repository Structure

```
stable_diffusion_carlos/
├── notebooks/
│   ├── 01_Raw_Ingredients.ipynb        # Images & text as numbers
│   ├── 02_The_Translator_CLIP.ipynb    # Contrastive learning & CLIP
│   ├── 03_The_Engine_Diffusion.ipynb   # Noise schedules & denoising
│   ├── 04_The_Conversation_CrossAttn.ipynb  # Cross-attention & guidance
│   └── 05_The_Road_Trip_Pipeline.ipynb # Full pipeline & evaluation
├── src/                                # Model architectures & training code
│   ├── models.py                       # TextEncoder, ImageEncoder, MicroUNet, etc.
│   ├── data.py                         # Shape dataset generation
│   ├── train.py                        # Training loops
│   ├── evaluate.py                     # Color/shape accuracy metrics
│   └── viz.py                          # Visualization helpers
├── assets/
│   ├── cheat_sheet_template.txt        # The Co-Pilot Reference Card
│   └── quiz_questions.json             # Pre-test and post-test questions
├── weights/
│   └── full_pipeline.pt                # Pre-trained weights (~1.5 MB)
├── tests/                              # Automated tests
└── pretrain.py                         # Script to regenerate weights
```

## License

MIT

## Acknowledgments

Course designed for the CAS BMAI program. Built using PyTorch, Matplotlib, and ipywidgets. All training data is generated programmatically — no external datasets required.
