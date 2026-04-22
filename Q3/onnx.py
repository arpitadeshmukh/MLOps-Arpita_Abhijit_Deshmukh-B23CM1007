"""
=============================================================================
Task 1: LoRA Fine-Tuning on Stable Diffusion v1-5 (Naruto Dataset)
Task 2: ONNX Export of the merged model
=============================================================================
Requirements:
    pip install torch torchvision diffusers transformers accelerate datasets
    pip install peft onnx onnxruntime pillow tqdm
=============================================================================
"""

import os
import math
import time
import torch
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID          = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASET_NAME      = "lambda/naruto-blip-captions"
OUTPUT_DIR        = "./sd_lora_output"
ONNX_DIR          = "./sd_onnx_output"
LORA_RANK         = 4          # r — decomposition rank
LORA_ALPHA        = 4          # scaling factor
NUM_TRAIN_EPOCHS  = 1
TRAIN_BATCH_SIZE  = 1
GRAD_ACCUM_STEPS  = 4
LEARNING_RATE     = 1e-4
MAX_TRAIN_STEPS   = 200        # keep short for demo; increase for better quality
IMAGE_SIZE        = 512
MIXED_PRECISION   = "fp16"     # or "no" for CPU
SEED              = 42

PROMPTS = [
    "Bill Gates with a hoodie",
    "John Oliver with Naruto style",
    "Hello Kitty with Naruto style",
    "Lebron James with a hat",
    "A photograph of an orange cat with Naruto style",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ONNX_DIR,   exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — LoRA Fine-Tuning
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(model):
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── 1a. Load base pipeline & report parameter count ──────────────────────────
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

print("\n" + "="*70)
print("TASK 1  —  LoRA Fine-Tuning")
print("="*70)

print("[INFO] Loading base Stable Diffusion v1-5 …")
tokenizer   = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder= CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
vae         = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
unet        = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
noise_sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# ── Count BASE model parameters ───────────────────────────────────────────────
unet_total,  _  = count_parameters(unet)
te_total,    _  = count_parameters(text_encoder)
vae_total,   _  = count_parameters(vae)
base_total       = unet_total + te_total + vae_total

print(f"\n[RESULT] Parameter counts (base model):")
print(f"         UNet          : {unet_total:,}")
print(f"         Text Encoder  : {te_total:,}")
print(f"         VAE           : {vae_total:,}")
print(f"  >>>  TOTAL BASE MODEL: {base_total:,}")


# ── 1b. Attach LoRA adapters to UNet attention layers ────────────────────────
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r              = LORA_RANK,
    lora_alpha     = LORA_ALPHA,
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"],  # attention projections
    lora_dropout   = 0.1,
    bias           = "none",
)

unet_lora = get_peft_model(unet, lora_config)

unet_lora_total,    unet_lora_trainable = count_parameters(unet_lora)
lora_only_params = unet_lora_trainable          # only LoRA layers are trainable

combined_total = base_total + lora_only_params  # conceptual: base + new adapters

print(f"\n[RESULT] LoRA adapter parameters:")
print(f"  >>>  TRAINABLE LoRA PARAMS : {lora_only_params:,}")
print(f"  >>>  COMBINED TOTAL (Base+LoRA): {base_total + lora_only_params:,}")
unet_lora.print_trainable_parameters()


# ── 1c. Load dataset ──────────────────────────────────────────────────────────
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

print("\n[INFO] Loading dataset …")
dataset = load_dataset(DATASET_NAME, split="train")

image_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def preprocess(examples):
    images  = [img.convert("RGB") for img in examples["image"]]
    pixel_values = [image_transforms(img) for img in images]
    captions     = examples["text"]
    return {"pixel_values": pixel_values, "captions": captions}

dataset = dataset.with_transform(preprocess)

def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    captions     = [e["captions"] for e in examples]
    return {"pixel_values": pixel_values, "captions": captions}

train_loader = DataLoader(
    dataset,
    batch_size  = TRAIN_BATCH_SIZE,
    shuffle     = True,
    collate_fn  = collate_fn,
    num_workers = 0,
)
print(f"[INFO] Dataset loaded — {len(dataset)} samples.")


# ── 1d. Training setup ────────────────────────────────────────────────────────
# Freeze everything except LoRA params
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Cast to appropriate dtype
weight_dtype = torch.float16 if (MIXED_PRECISION == "fp16" and device == "cuda") else torch.float32

vae.to(device, dtype=weight_dtype)
text_encoder.to(device, dtype=weight_dtype)
unet_lora.to(device)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet_lora.parameters()),
    lr=LEARNING_RATE,
)

scaler = torch.cuda.amp.GradScaler() if (MIXED_PRECISION == "fp16" and device == "cuda") else None


# ── 1e. Training loop ─────────────────────────────────────────────────────────
print(f"\n[INFO] Starting training for up to {MAX_TRAIN_STEPS} steps …")

global_step = 0
final_loss  = None
unet_lora.train()

epoch_losses = []
start_time   = time.time()

for epoch in range(NUM_TRAIN_EPOCHS):
    for step, batch in enumerate(train_loader):
        if global_step >= MAX_TRAIN_STEPS:
            break

        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        captions     = batch["captions"]

        # Encode images → latent space
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # Sample noise & timestep
        noise      = torch.randn_like(latents)
        bsz        = latents.shape[0]
        timesteps  = torch.randint(0, noise_sched.config.num_train_timesteps,
                                   (bsz,), device=device).long()
        noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

        # Encode text
        with torch.no_grad():
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            encoder_hidden_states = text_encoder(
                text_inputs.input_ids.to(device)
            )[0]

        # Forward pass
        if scaler:
            with torch.cuda.amp.autocast():
                noise_pred = unet_lora(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(
                    noise_pred.float(), noise.float(), reduction="mean"
                )
        else:
            noise_pred = unet_lora(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(
                noise_pred.float(), noise.float(), reduction="mean"
            )

        loss_val = loss.item()
        epoch_losses.append(loss_val)
        final_loss = loss_val

        # Backward
        if scaler:
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, unet_lora.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, unet_lora.parameters()), 1.0
                )
                optimizer.step()
                optimizer.zero_grad()

        global_step += 1
        if global_step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {global_step:4d}/{MAX_TRAIN_STEPS}  |  loss = {loss_val:.6f}  |  elapsed = {elapsed:.1f}s")

    if global_step >= MAX_TRAIN_STEPS:
        break

print(f"\n[RESULT] >>>  FINAL TRAINING LOSS : {final_loss:.6f}")


# ── 1f. Save LoRA adapter weights ─────────────────────────────────────────────
lora_save_path = os.path.join(OUTPUT_DIR, "lora_adapter")
unet_lora.save_pretrained(lora_save_path)
print(f"[INFO] LoRA adapter saved → {lora_save_path}")

# Calculate file size of saved LoRA weights
lora_bytes = sum(
    f.stat().st_size for f in Path(lora_save_path).rglob("*") if f.is_file()
)
lora_mb = lora_bytes / (1024 ** 2)
print(f"\n[RESULT] >>>  LoRA adapter file size : {lora_mb:.2f} MB")


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — ONNX Export
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 2  —  ONNX Model Export")
print("="*70)

# ── 2a. Merge LoRA weights into base UNet ────────────────────────────────────
from peft import PeftModel

print("[INFO] Merging LoRA weights into base UNet …")
unet_merged = unet_lora.merge_and_unload()   # returns a plain nn.Module
unet_merged.eval()
print("[INFO] Merge complete.")

# ── 2b. Measure baseline model size on disk ──────────────────────────────────
import huggingface_hub, shutil

# Download model snapshot to get disk size
print("[INFO] Calculating baseline model disk size (downloading snapshot) …")
try:
    snapshot_dir = huggingface_hub.snapshot_download(MODEL_ID, ignore_patterns=["*.msgpack","*.h5"])
    base_bytes   = sum(f.stat().st_size for f in Path(snapshot_dir).rglob("*") if f.is_file())
    base_gb      = base_bytes / (1024 ** 3)
    print(f"\n[RESULT] >>>  ORIGINAL BASELINE MODEL SIZE : {base_gb:.2f} GB")
except Exception as e:
    print(f"[WARN] Could not measure snapshot: {e}")
    base_gb = None


# ── 2c. Export Text Encoder to ONNX ──────────────────────────────────────────
import torch.onnx

print("\n[INFO] Exporting Text Encoder to ONNX …")
text_encoder.to("cpu").eval()
te_onnx_path = os.path.join(ONNX_DIR, "text_encoder.onnx")

dummy_input_ids = torch.zeros(1, tokenizer.model_max_length, dtype=torch.long)

with torch.no_grad():
    torch.onnx.export(
        text_encoder,
        dummy_input_ids,
        te_onnx_path,
        input_names  = ["input_ids"],
        output_names = ["last_hidden_state", "pooler_output"],
        dynamic_axes = {
            "input_ids"         : {0: "batch"},
            "last_hidden_state" : {0: "batch"},
        },
        opset_version = 14,
        do_constant_folding=True,
    )
print(f"[INFO] Text Encoder ONNX → {te_onnx_path}")


# ── 2d. Export UNet to ONNX ───────────────────────────────────────────────────
print("[INFO] Exporting UNet to ONNX …")
unet_merged.to("cpu").eval()
unet_onnx_path = os.path.join(ONNX_DIR, "unet.onnx")

latent_channels = unet_merged.config.in_channels  # 4
latent_size     = IMAGE_SIZE // 8                 # 64

dummy_latents   = torch.randn(1, latent_channels, latent_size, latent_size)
dummy_timestep  = torch.tensor([1])
dummy_enc_hs    = torch.randn(1, tokenizer.model_max_length, text_encoder.config.hidden_size)

with torch.no_grad():
    torch.onnx.export(
        unet_merged,
        (dummy_latents, dummy_timestep, dummy_enc_hs),
        unet_onnx_path,
        input_names  = ["latent_model_input", "timestep", "encoder_hidden_states"],
        output_names = ["noise_pred"],
        dynamic_axes = {
            "latent_model_input"   : {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "noise_pred"           : {0: "batch"},
        },
        opset_version = 14,
        do_constant_folding=True,
    )
print(f"[INFO] UNet ONNX → {unet_onnx_path}")


# ── 2e. Export VAE Decoder to ONNX ───────────────────────────────────────────
print("[INFO] Exporting VAE Decoder to ONNX …")
vae_decoder = vae.decoder
vae_decoder.to("cpu").eval()
vae_onnx_path = os.path.join(ONNX_DIR, "vae_decoder.onnx")

dummy_vae_input = torch.randn(1, 4, latent_size, latent_size)

with torch.no_grad():
    torch.onnx.export(
        vae_decoder,
        dummy_vae_input,
        vae_onnx_path,
        input_names  = ["latent_sample"],
        output_names = ["decoded_image"],
        dynamic_axes = {
            "latent_sample"  : {0: "batch"},
            "decoded_image"  : {0: "batch"},
        },
        opset_version = 14,
        do_constant_folding=True,
    )
print(f"[INFO] VAE Decoder ONNX → {vae_onnx_path}")


# ── 2f. Measure ONNX model sizes ─────────────────────────────────────────────
def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)

te_mb   = file_size_mb(te_onnx_path)
unet_mb = file_size_mb(unet_onnx_path)
vae_mb  = file_size_mb(vae_onnx_path)
total_onnx_gb = (te_mb + unet_mb + vae_mb) / 1024

print(f"\n[RESULT] ONNX file sizes:")
print(f"         Text Encoder : {te_mb:.2f} MB")
print(f"         UNet         : {unet_mb:.2f} MB")
print(f"         VAE Decoder  : {vae_mb:.2f} MB")
print(f"  >>>  COMBINED ONNX SIZE: {total_onnx_gb:.3f} GB")


# ── 2g. Verify ONNX graphs load correctly ────────────────────────────────────
import onnxruntime as ort

print("\n[INFO] Verifying ONNX models with ONNX Runtime …")
for name, path in [("Text Encoder", te_onnx_path),
                   ("UNet",         unet_onnx_path),
                   ("VAE Decoder",  vae_onnx_path)]:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inputs  = [i.name for i in sess.get_inputs()]
    outputs = [o.name for o in sess.get_outputs()]
    print(f"  ✓  {name:15s}  inputs={inputs}  outputs={outputs}")

print("\n[INFO] All ONNX models verified successfully.")


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE — Generate images with the fine-tuned model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("INFERENCE — Generating images with fine-tuned LoRA model")
print("="*70)

from diffusers import StableDiffusionPipeline
from peft import PeftModel

print("[INFO] Building inference pipeline with merged weights …")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    unet      = unet_merged.to(device),
    torch_dtype = weight_dtype if device == "cuda" else torch.float32,
    safety_checker = None,
)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)

generator = torch.Generator(device=device).manual_seed(SEED)

output_images_dir = os.path.join(OUTPUT_DIR, "generated_images")
os.makedirs(output_images_dir, exist_ok=True)

for idx, prompt in enumerate(PROMPTS):
    print(f"  Generating: {prompt}")
    result = pipe(
        prompt,
        num_inference_steps = 30,
        guidance_scale      = 7.5,
        generator           = generator,
    )
    img_path = os.path.join(output_images_dir, f"output_{idx+1}.png")
    result.images[0].save(img_path)
    print(f"    → Saved: {img_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)
print(f"  Total Base Model Parameters          : {base_total:,}")
print(f"  Trainable LoRA Parameters            : {lora_only_params:,}")
print(f"  Combined (Base + LoRA) Parameters    : {base_total + lora_only_params:,}")
print(f"  Final Training Loss                  : {final_loss:.6f}")
print(f"  LoRA Adapter File Size               : {lora_mb:.2f} MB")
if base_gb:
    print(f"  Original Baseline Model Size         : {base_gb:.2f} GB")
print(f"  Combined ONNX Export Size            : {total_onnx_gb:.3f} GB")
print("="*70)
print("[DONE] All tasks completed successfully.")