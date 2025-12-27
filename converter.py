import argparse
import torch
import numpy as np
from safetensors.numpy import save_file
from tqdm import tqdm

def map_key_wan_video(key):
    # Logic mapping dựa trên WanModelStateDictConverter từ code gốc
    rename_dict = {
        "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
        "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
        "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
        "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
        "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
        "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
        "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
        "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
        "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
        "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
        "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
        "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
        "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
        "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
        "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
        "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
        "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
        "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
        "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
        "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
        "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
        "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
        "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
        "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
        "blocks.0.norm2.bias": "blocks.0.norm3.bias",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "blocks.0.scale_shift_table": "blocks.0.modulation",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "scale_shift_table": "head.modulation",
        "proj_out.bias": "head.head.bias",
        "proj_out.weight": "head.head.weight",
    }

    if key in rename_dict:
        return rename_dict[key]
    
    # Handle pattern mapping (blocks.X...)
    parts = key.split(".")
    if len(parts) > 2 and parts[0] == "blocks":
        # Create a synthetic key for block 0 to check against dict
        # e.g., blocks.5.attn1.to_q.weight -> blocks.0.attn1.to_q.weight
        synthetic_key = ".".join(["blocks", "0"] + parts[2:])
        if synthetic_key in rename_dict:
            mapped_suffix = rename_dict[synthetic_key]
            # Replace '0' back to original index
            mapped_parts = mapped_suffix.split(".")
            mapped_parts[1] = parts[1]
            return ".".join(mapped_parts)
            
    return key

def convert(ckpt_path, output_path, dtype=np.float16):
    print(f"Loading {ckpt_path}...")
    sd = torch.load(ckpt_path, map_location="cpu")
    if "model_state" in sd:
        sd = sd["model_state"] # Handle wrapper if present

    mlx_sd = {}
    
    print("Converting weights...")
    for k, v in tqdm(sd.items()):
        new_k = map_key_wan_video(k)
        
        # Convert tensor to numpy
        if isinstance(v, torch.Tensor):
            val = v.detach().float().numpy() # Process in float32 then cast later
        else:
            val = v

        # Shape Transformation for MLX (Channel-Last)
        
        # 1. Linear Layers: PyTorch (Out, In) -> MLX (Out, In) but stored as (In, Out) for x@W
        #    Wait, standard MLX load matches PyTorch if we transpose?
        #    MLX nn.Linear(input_dim, output_dim) creates weight (input_dim, output_dim).
        #    PyTorch is (output_dim, input_dim).
        #    So we strictly need to TRANSPOSE (1, 0).
        if len(val.shape) == 2 and "weight" in new_k and "embedding" not in new_k and "modulation" not in new_k:
             # Exclude embeddings (N, D) which are same.
             # Exclude modulation parameters if they are treated as embeddings/params (check shape).
             # Usually 'modulation' in DiT is (1, 6, dim) or (1, 2, dim), handled separately.
             if "time_embedding" in new_k or "text_embedding" in new_k or "head.head" in new_k or "ffn" in new_k or "to_q" in new_k or "to_k" in new_k or "to_v" in new_k or "to_out" in new_k or "time_proj" in new_k:
                 val = val.transpose(1, 0)

        # 2. Conv3D: PyTorch (Out, In, T, H, W) -> MLX (Out, T, H, W, In)
        #    MLX nn.Conv3d weight expected shape: (out_channels, kernel_depth, kernel_height, kernel_width, in_channels)
        if len(val.shape) == 5:
            # PyTorch: (Out, In, D, H, W) -> (0, 1, 2, 3, 4)
            # MLX:     (Out, D, H, W, In) -> (0, 2, 3, 4, 1)
            val = val.transpose(0, 2, 3, 4, 1)

        # 3. Conv2D: PyTorch (Out, In, H, W) -> MLX (Out, H, W, In)
        if len(val.shape) == 4:
            # PyTorch: (Out, In, H, W) -> (0, 1, 2, 3)
            # MLX:     (Out, H, W, In) -> (0, 2, 3, 1)
            val = val.transpose(0, 2, 3, 1)

        mlx_sd[new_k] = val.astype(dtype)

    print(f"Saving to {output_path}...")
    save_file(mlx_sd, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to PyTorch .pth or .ckpt")
    parser.add_argument("--out", type=str, required=True, help="Path to Output .safetensors")
    args = parser.parse_args()
    convert(args.ckpt, args.out)
