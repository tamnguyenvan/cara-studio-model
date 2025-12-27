import argparse
import os
import sys
import time
import mlx.core as mx
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm

# Import các module đã port
# Giả sử bạn gộp WanModel, WanVideoVAE, Buffer_LQ4x_Proj, FlashVSRPipeline vào flashvsr_mlx.py
from flashvsr_mlx import (
    WanModel, 
    WanVideoVAE, 
    Buffer_LQ4x_Proj, 
    FlashVSRPipeline
)

# ==============================================================================
# Configuration (FlashVSR Tiny)
# ==============================================================================
# Hash: 9269f8db9040a9d860eaca435be61814 (Từ code gốc)
FLASHVSR_TINY_CONFIG = {
    "model_type": "t2v",
    "patch_size": (1, 2, 2),
    "in_dim": 16,        # Latent channels
    "dim": 1536,         # Hidden dim
    "ffn_dim": 8960,     # MLP hidden dim
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 12,
    "num_layers": 30,
    "eps": 1e-6,
}

# ==============================================================================
# Helper Functions
# ==============================================================================

def load_weights_safetensors(model, path):
    print(f"Loading weights from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")
    
    # Load dictionary of arrays
    weights = mx.load(path)
    
    # Apply to model
    # strict=False giúp tránh lỗi nếu file weight chứa thừa tham số (VD: training stats)
    model.load_weights(list(weights.items()), strict=False)
    print("Weights loaded successfully.")

def load_video(path, max_frames=None, height=None, width=None):
    print(f"Reading video from {path}...")
    reader = imageio.get_reader(path)
    fps = reader.get_meta_data()['fps']
    
    frames = []
    for i, frame in enumerate(reader):
        if max_frames and i >= max_frames:
            break
            
        # Resize if requested (Simple PIL resize)
        if height and width:
            img = Image.fromarray(frame)
            img = img.resize((width, height), Image.BICUBIC)
            frame = np.array(img)
            
        # Normalize [0, 1] and convert to Float32
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
        
    reader.close()
    
    # Stack -> (T, H, W, C)
    video_tensor = np.stack(frames)
    
    # Convert to MLX
    return mx.array(video_tensor), fps

def save_video(tensor, path, fps=30):
    # tensor: (T, H, W, C) range [0, 1]
    print(f"Saving video to {path}...")
    
    # Denormalize -> [0, 255] -> uint8
    frames = np.array(tensor * 255.0).clip(0, 255).astype(np.uint8)
    
    writer = imageio.get_writer(path, fps=fps, quality=6) # quality=6 ~ crf 20-23
    for frame in tqdm(frames, desc="Writing frames"):
        writer.append_data(frame)
    writer.close()

def load_torch_prompt(path):
    """
    Load posi_prompt.pth (PyTorch Tensor) và chuyển sang MLX.
    Cần cài torch chỉ để load file này nếu chưa convert.
    """
    print(f"Loading context/prompt from {path}...")
    try:
        import torch
        # Load on CPU
        ctx = torch.load(path, map_location="cpu")
        # Convert to numpy then MLX
        return mx.array(ctx.float().numpy())
    except ImportError:
        print("Error: PyTorch not installed. Please convert 'posi_prompt.pth' to .npy first.")
        sys.exit(1)

class ModelManager:
    def __init__(self):
        self.dit = None
        self.vae = None
        self.lq_proj = None

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="FlashVSR MLX Inference")
    
    # Inputs
    parser.add_argument("--input", type=str, required=True, help="Path to input LQ video")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output video")
    
    # Model Weights (Converted Safetensors)
    parser.add_argument("--dit_ckpt", type=str, required=True, help="Path to DiT .safetensors")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to VAE .safetensors")
    parser.add_argument("--lq_ckpt", type=str, required=True, help="Path to LQ Proj .safetensors")
    parser.add_argument("--prompt_path", type=str, default="posi_prompt.pth", help="Path to posi_prompt.pth (context)")
    
    # Parameters
    parser.add_argument("--num_frames", type=int, default=None, help="Limit number of frames")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile_vae", action="store_true", default=True, help="Use tiling for VAE (saves RAM)")
    
    args = parser.parse_args()

    # 1. Initialize Models
    print("Initializing models...")
    manager = ModelManager()
    
    # DiT
    manager.dit = WanModel(FLASHVSR_TINY_CONFIG)
    load_weights_safetensors(manager.dit, args.dit_ckpt)
    
    # VAE
    manager.vae = WanVideoVAE(z_dim=16, dim=96)
    load_weights_safetensors(manager.vae, args.vae_ckpt)
    
    # LQ Projector (Feature Extractor)
    # Output dim must match DiT dim (1536)
    manager.lq_proj = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536)
    load_weights_safetensors(manager.lq_proj, args.lq_ckpt)
    
    # Quantization (Optional optimization)
    # Nếu RAM ít (8GB/16GB), uncomment dòng dưới để nén Linear layers của DiT xuống 4-bit/8-bit
    # import mlx.nn as nn
    nn.QuantizedLinear.quantize_module(manager.dit, group_size=32, bits=4)
    print("Quantized DiT to 4-bit.")

    # 2. Setup Pipeline
    pipe = FlashVSRPipeline(manager)
    
    # Load Context (Prompt Embedding)
    if args.prompt_path.endswith(".pth"):
        context = load_torch_prompt(args.prompt_path)
    else:
        # Assume .npy or .npz
        context = mx.array(np.load(args.prompt_path))
        
    # Inject Context into DiT Cross-Attention
    pipe.init_cross_kv(context)

    # 3. Process Video
    lq_video, fps = load_video(args.input, max_frames=args.num_frames)
    B, T, H, W, C = 1, lq_video.shape[0], lq_video.shape[1], lq_video.shape[2], lq_video.shape[3]
    
    print(f"Input Video: {T} frames, {H}x{W}")
    
    # Add Batch Dimension (B=1)
    lq_video_batch = lq_video[None, ...] # (1, T, H, W, C)

    # 4. Run Inference
    # Lưu ý: FlashVSR VAE upsampling x8 spatial. 
    # Nếu input 480p, output sẽ rất lớn. Code gốc thường upscale LQ trước rồi đưa vào.
    # Tuy nhiên Pipeline FlashVSR Tiny thường nhận LQ native và trả về HQ x4.
    
    output_video = pipe(
        LQ_video=lq_video_batch,
        seed=args.seed,
        tiled_vae=args.tile_vae,
        color_fix=True # Bật color correction
    )
    
    # 5. Save
    save_video(output_video, args.output, fps=fps)
    print("Done!")

if __name__ == "__main__":
    main()
