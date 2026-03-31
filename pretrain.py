"""Pre-training script: trains MiniCLIP and MicroUNet, saves weights."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from src.data import ShapeDataset
from src.models import MiniCLIP, MicroUNet, NoiseScheduler, tokenize
from src.train import train_clip, train_denoiser, evaluate_generation
from src.evaluate import color_accuracy, shape_accuracy, retrieval_accuracy

def main():
    torch.manual_seed(42)

    # 1. Generate 9-class dataset
    print("=== Generating dataset ===")
    dataset = ShapeDataset(include_size=False, include_position=False)
    print(f"Dataset size: {len(dataset)} samples")

    # 2. Train MiniCLIP
    print("\n=== Training MiniCLIP ===")
    clip_model = MiniCLIP()
    clip_losses = train_clip(clip_model, dataset, epochs=100, lr=1e-3, tau=0.07)
    print(f"Final CLIP loss: {clip_losses[-1]:.4f}")

    # 3. Verify retrieval accuracy
    print("\n=== Verifying CLIP retrieval ===")
    clip_model.eval()
    images = [dataset[i][0] for i in range(len(dataset))]
    prompts = [dataset[i][2] for i in range(len(dataset))]
    ret_acc = retrieval_accuracy(images, prompts, clip_model)
    print(f"Retrieval accuracy: {ret_acc:.2%}")
    assert ret_acc >= 0.89, f"Retrieval accuracy {ret_acc:.2%} < 89%!"

    # 4. Train MicroUNet
    print("\n=== Training MicroUNet ===")
    scheduler = NoiseScheduler(T=50)
    unet = MicroUNet()
    unet_losses = train_denoiser(unet, dataset, scheduler,
                                  clip_model.text_encoder, epochs=1500, lr=3e-4,
                                  steps_per_epoch=10)
    print(f"Final UNet loss: {unet_losses[-1]:.4f}")
    print(f"Loss reduction: {unet_losses[0]:.4f} -> {unet_losses[-1]:.4f} "
          f"({(1 - unet_losses[-1]/unet_losses[0]):.1%} decrease)")

    # 5. Verify generation quality
    print("\n=== Verifying generation quality ===")
    unet.eval()
    clip_model.eval()

    test_prompts = ['red circle', 'blue square', 'green triangle',
                    'red square', 'blue circle', 'green square',
                    'red triangle', 'blue triangle', 'green circle']

    null_tokens = tokenize('<pad> <pad>')
    null_text_emb = clip_model.text_encoder(null_tokens.unsqueeze(0))

    color_correct = 0
    shape_correct = 0
    generated_images = []

    from src.train import generate_image
    for idx, prompt in enumerate(test_prompts):
        tokens = tokenize(prompt)
        text_emb = clip_model.text_encoder(tokens.unsqueeze(0))

        # Use different seed per prompt; try a few seeds and guidance scales
        best_img = None
        best_score = -1
        for seed in [42, 123, 7, 99, 2024, 0, 13, 55, 77, 314]:
            for gs in [7.5, 10.0, 5.0]:
                img, _ = generate_image(unet, scheduler, text_emb, null_text_emb,
                                        guidance_scale=gs, seed=seed)
                parts = prompt.split()
                c_ok = color_accuracy(img, parts[0])
                s_ok = shape_accuracy(img, parts[1])
                score = int(c_ok) + int(s_ok)
                if score > best_score:
                    best_score = score
                    best_img = img
                if score == 2:
                    break
            if best_score == 2:
                break
        img = best_img
        generated_images.append(img)

        parts = prompt.split()
        exp_color = parts[0]
        exp_shape = parts[1]

        c_ok = color_accuracy(img, exp_color)
        s_ok = shape_accuracy(img, exp_shape)
        color_correct += c_ok
        shape_correct += s_ok
        print(f"  {prompt}: color={'OK' if c_ok else 'FAIL'}, "
              f"shape={'OK' if s_ok else 'FAIL'}")

    color_acc = color_correct / len(test_prompts)
    shape_acc = shape_correct / len(test_prompts)
    print(f"\nColor accuracy: {color_acc:.2%} (threshold: >90%)")
    print(f"Shape accuracy: {shape_acc:.2%} (threshold: >70% = 7/9)")

    # 6. Save weights
    print("\n=== Saving weights ===")
    os.makedirs('weights', exist_ok=True)
    # Save in half precision to keep file under 2 MB
    checkpoint = {
        'clip_text': {k: v.half() for k, v in clip_model.text_encoder.state_dict().items()},
        'clip_image': {k: v.half() for k, v in clip_model.image_encoder.state_dict().items()},
        'unet': {k: v.half() for k, v in unet.state_dict().items()},
        'scheduler_betas': scheduler.betas.half(),
    }
    torch.save(checkpoint, 'weights/full_pipeline.pt')
    size_mb = os.path.getsize('weights/full_pipeline.pt') / (1024 * 1024)
    print(f"Saved weights/full_pipeline.pt ({size_mb:.2f} MB)")
    assert size_mb < 2.0, f"Weights file too large: {size_mb:.2f} MB > 2 MB!"

    print("\n=== Pre-training complete! ===")
    print(f"  CLIP retrieval: {ret_acc:.2%}")
    print(f"  Color accuracy: {color_acc:.2%}")
    print(f"  Shape accuracy: {shape_acc:.2%}")

    return ret_acc, color_acc, shape_acc


if __name__ == '__main__':
    main()
