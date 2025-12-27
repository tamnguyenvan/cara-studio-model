import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, List

# Re-using RMSNorm and CausalConv3dZeroPad from flashvsr_mlx.py

# ==============================================================================
# VAE Encoder/Decoder Components
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False), # Assuming images=False for channel-last
            nn.SiLU(),
            CausalConv3dZeroPad(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3dZeroPad(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3dZeroPad(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def __call__(self, x, feat_cache=None, feat_idx=[0]):
        # x: (B, T, H, W, C)
        h = self.shortcut(x)
        
        for layer in self.residual:
            if isinstance(layer, CausalConv3dZeroPad) and feat_cache is not None:
                # Handle cache logic from original code
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :, :].copy() # Cache last CACHE_T frames
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
                
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim, images=False)
        # Conv2d layers expect (B, C, T, H, W) in MLX
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1) 
        self.proj = nn.Conv2d(dim, dim, 1)
        # Initialize proj weight to zeros
        self.proj.weight = mx.zeros_like(self.proj.weight)

    def __call__(self, x):
        # x: (B, T, H, W, C)
        identity = x
        B, T, H, W, C = x.shape
        
        # Reshape for Conv2d: (B, T, H, W, C) -> (B*T, C, H, W)
        x_reshaped = x.transpose(0, 4, 1, 2, 3) # (B, C, T, H, W)
        
        x_reshaped = self.norm(x_reshaped)
        qkv = self.to_qkv(x_reshaped) # (B*T, 3*C, H, W)
        
        # Split q, k, v. Reshape for attention (B_eff, N_heads, SeqLen, Head_Dim)
        # This part assumes SDPA is available and handles Batching.
        # For simplicity, let's assume it's dense attention for now.
        # The original code uses F.scaled_dot_product_attention which expects (N, S, D) for QKV
        # Let's re-map to (N_eff, S, D) -> (B*T, H*W, C) for Conv2d output
        
        # If it's Spatial Attention on each frame independently:
        qkv_flat = qkv.transpose(0, 2, 3, 1).reshape(B*T, H*W, 3*C) # (B*T, HW, 3C)
        q, k, v = mx.split(qkv_flat, 3, axis=-1) # Each is (B*T, HW, C)
        
        # Use MLX SDPA (assuming no explicit masks here for attention block)
        # Need to know num_heads for SDPA. Assuming C is C*1 (1 head) or C is split.
        # If C is the total dim, we need num_heads. Let's assume num_heads = 1 implicitly for this layer.
        # The original code does q.reshape(B*t, 1, c*3, -1).permute(0,1,3,2).chunk(3,-1)
        # This seems to flatten spatial dims into seq length for attention.
        
        # Simplified: Assuming C is total dim, num_heads = 1
        # SDPA expects (batch, heads, seq, dim_head)
        q = q.reshape(B*T, 1, H*W, C)
        k = k.reshape(B*T, 1, H*W, C)
        v = v.reshape(B*T, 1, H*W, C)
        
        attn_output = mx.fast.scaled_dot_product_attention(q, k, v) # (B*T, 1, H*W, C)
        attn_output = attn_output.squeeze(1).reshape(B*T, H*W, C) # (B*T, HW, C)
        
        # Reshape back: (B*T, HW, C) -> (B*T, C, H, W)
        attn_output = attn_output.reshape(B*T, H, W, C).transpose(0, 3, 1, 2) # (B*T, C, H, W)
        
        attn_output = self.proj(attn_output)
        
        # Reshape back to original video format: (B, T, H, W, C)
        attn_output = attn_output.transpose(0, 2, 3, 4, 1) # (B, T, H, W, C)
        
        return identity + attn_output

class Encoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[True, True, False], dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        
        dims = [dim * u for u in [1] + dim_mult] # [128, 256, 512, 1024] if dim=128, dim_mult=[1,2,4,4]
        scale = 1.0
        
        # Initial Conv3d (Input: B, T, H, W, C_in)
        self.conv1 = CausalConv3dZeroPad(3, dims[0], 3, padding=1) # Input C=3, Output dim[0]
        
        self.downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales: self.downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim # Update in_dim for next block
            
            if i != len(dim_mult) - 1: # Not the last downsampling stage
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.downsamples.append(Resample(out_dim, mode=mode)) # Resample handles channel reduction internally
                scale /= 2.0 # Scale reduces for subsequent layers
                
        self.downsamples = nn.Sequential(*self.downsamples) # Convert list to Sequential
        
        # Middle block (after last downsampling)
        final_downsample_dim = dims[-1]
        self.middle = nn.Sequential(
            ResidualBlock(final_downsample_dim, final_downsample_dim, dropout),
            AttentionBlock(final_downsample_dim),
            ResidualBlock(final_downsample_dim, final_downsample_dim, dropout)
        )
        
        # Head: Final Conv3d to latent space (z_dim * 2 for mu/logvar)
        self.head = nn.Sequential(
            RMSNorm(final_downsample_dim, images=False),
            nn.SiLU(),
            CausalConv3dZeroPad(final_downsample_dim, z_dim * 2, 3, padding=1) # Output is mu & logvar
        )
    
    def __call__(self, x, feat_cache=None, feat_idx=[0]):
        # x: (B, T, H, W, C_in=3)
        
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :].copy() # Cache last CACHE_T frames
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x) # Initial Conv3d
            
        # Downsampling blocks
        for layer in self.downsamples:
            if isinstance(layer, (ResidualBlock, AttentionBlock)) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            elif isinstance(layer, Resample) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx) # Resample might also need cache
            else:
                x = layer(x)
                
        # Middle block
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
                
        # Head
        for layer in self.head:
            if isinstance(layer, CausalConv3dZeroPad) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :, :].copy()
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x # Output (B, T, H, W, z_dim*2) for mu, logvar

class Decoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_upsample=[False, True, True], dropout=0.0):
        super().__init__()
        self.dim, self.z_dim, self.dim_mult, self.num_res_blocks, self.attn_scales, self.temperal_upsample = dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_upsample
        self.temperal_upsample = temperal_upsample[::-1] # Reverse for decoder order
        
        # Latent dim is z_dim * 2 (mu, logvar) for encoder, but only z_dim for decoder input
        # Here z_dim refers to the latent space dim of the decoder input
        
        # dims calculation: Use dim_mult in reverse
        # Example: dim=128, dim_mult=[1,2,4,4] -> [128, 256, 512, 1024]
        # Decoder dims: [1024, 512, 256, 128] (reversed and shifted)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]] 
        
        # Scale calculation for Resample modes (upsampling)
        scale = 1.0 / 2 ** (len(dim_mult) - 2) # Base scale for first upsample
        
        # Initial Conv3d for decoder input (z_dim)
        self.conv1 = CausalConv3dZeroPad(z_dim, dims[0], 3, padding=1)
        
        # Middle block (same structure as encoder)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout)
        )
        
        self.upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Adjust in_dim for residual connections in decoder stages if channels doubled
            # Original code has `if i == 1 or i == 2 or i == 3: in_dim = in_dim // 2`
            # This implies some stages might reduce channels before block processing.
            # Let's mirror that logic:
            if i > 0: in_dim = in_dim // 2 # For stages 1, 2, 3 (i=0 is first stage)
            
            for _ in range(num_res_blocks + 1): # More ResBlocks in decoder usually
                self.upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales: self.upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim # Update for next block
            
            if i != len(dim_mult) - 1: # Not the last stage
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                self.upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0 # Scale increases for upsampling
                
        self.upsamples = nn.Sequential(*self.upsamples)
        
        # Head: Final Conv3d to output (3 channels, RGB)
        final_upsample_dim = dims[-1] // 2 # Final dim reduction
        self.head = nn.Sequential(
            RMSNorm(final_upsample_dim, images=False),
            nn.SiLU(),
            CausalConv3dZeroPad(final_upsample_dim, 3, 3, padding=1) # Output RGB (3 channels)
        )

    def __call__(self, x, feat_cache=None, feat_idx=[0]):
        # x: (B, T, H, W, C_in=z_dim)
        
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :].copy()
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
            
        # Middle block
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
                
        # Upsampling blocks
        for layer in self.upsamples:
            if isinstance(layer, (ResidualBlock, AttentionBlock)) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            elif isinstance(layer, Resample) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
                
        # Head
        for layer in self.head:
            if isinstance(layer, CausalConv3dZeroPad) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :, :].copy()
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x # Output (B, T_out, H_out, W_out, 3)

class VideoVAE_(nn.Module):
    def __init__(self, dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0):
        super().__init__()
        self.dim, self.z_dim, self.dim_mult, self.num_res_blocks, self.attn_scales, self.temperal_downsample = dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_downsample
        
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout)
        
        # Conv layers for mu/logvar and z -> initial decoder state
        self.conv1 = CausalConv3dZeroPad(z_dim * 2, z_dim * 2, 1) # Input channels = 2*z_dim (mu, logvar)
        self.conv2 = CausalConv3dZeroPad(z_dim, z_dim, 1) # Input channels = z_dim
        
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_upsample=[False, True, True][::-1], dropout=dropout)

        # Cache management for Conv layers
        self._conv_num = sum(1 for m in self.decoder.modules() if isinstance(m, CausalConv3dZeroPad))
        self._conv_idx = [0] # Global index for cache
        self._feat_map = [None] * self._conv_num # List to store cache tensors
        
        self._enc_conv_num = sum(1 for m in self.encoder.modules() if isinstance(m, CausalConv3dZeroPad))
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
        
        self.upsampling_factor = 8 # Based on VAE architecture (T, H, W)

    def encode(self, x, scale):
        # x: (B, T, H, W, C_in=3)
        # scale: list or tensor for mean/std scaling
        self.clear_cache() # Reset cache for new encode
        
        # Determine number of encoding steps based on temporal resolution
        # Original code: iter_ = 1 + (t - 1) // 4
        # Let's assume T is time dimension. This implies downsampling temporal by 4.
        T = x.shape[1]
        iter_ = 1 + (T - 1) // 4
        
        encoded_outputs = []
        for i in range(iter_):
            # Slice temporal dimension for encoding (step size of 4)
            start_t = 0 if i == 0 else 1 + 4 * (i - 1)
            end_t = 1 + 4 * i if i > 0 else 1
            
            clip_x = x[:, start_t:end_t, :, :, :]
            
            # Call encoder, passing cache
            out = self.encoder(clip_x, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            encoded_outputs.append(out)
            
        # Concatenate outputs along temporal dimension
        out = mx.concatenate(encoded_outputs, axis=1) # (B, T_encoded, H, W, C)
        
        # Split into mu and log_var
        mu, log_var = mx.split(self.conv1(out), 2, axis=-1)
        
        # Apply scaling if provided
        if isinstance(scale[0], mx.array): # If scale is a tensor/list
            scale_mean, scale_std = scale[0], scale[1]
            mu = (mu - scale_mean) * scale_std
        else:
            scale_mean, scale_std = scale[0], scale[1]
            mu = (mu - scale_mean) * scale_std
            
        return mu

    def decode(self, z, scale):
        # z: (B, T_latent, H_latent, W_latent, C_z) - encoded latents
        # scale: list or tensor for mean/std scaling
        self.clear_cache() # Reset cache for new decode
        
        # Apply inverse scaling
        if isinstance(scale[0], mx.array):
            scale_mean, scale_std = scale[0], scale[1]
            z = z / scale_std + scale_mean
        else:
            scale_mean, scale_std = scale[0], scale[1]
            z = z / scale_std + scale_mean
            
        # Initial Conv3d for decoder input
        x = self.conv2(z)
        
        # Determine number of decoding steps based on temporal resolution of latents
        T_latent = x.shape[1]
        iter_ = T_latent # Each temporal slice is processed independently by decoder conv blocks
        
        decoded_outputs = []
        for i in range(iter_):
            # Slice temporal dimension (decoder processes one frame/slice at a time)
            clip_x = x[:, i:i+1, :, :, :]
            
            # Call decoder, passing cache
            out = self.decoder(clip_x, feat_cache=self._feat_map, feat_idx=self._conv_idx)
            decoded_outputs.append(out)
            
        # Concatenate decoded frames along temporal dimension
        out = mx.concatenate(decoded_outputs, axis=1) # (B, T_output, H, W, 3)
        return out

    def stream_decode(self, z, scale):
        # z: (B, T_latent, H_latent, W_latent, C_z) - encoded latents for one frame/slice
        # scale: list or tensor
        # This method expects z to be a single temporal slice (T_latent=1) and manages its own cache.
        
        # Apply inverse scaling
        if isinstance(scale[0], mx.array):
            scale_mean, scale_std = scale[0], scale[1]
            z = z / scale_std + scale_mean
        else:
            scale_mean, scale_std = scale[0], scale[1]
            z = z / scale_std + scale_mean
        
        # Initial Conv3d for decoder input
        x = self.conv2(z) # Input z is already (B, 1, H, W, C_z) if T_latent=1
        
        # Call decoder for this single slice
        out = self.decoder(x, feat_cache=self._feat_map, feat_idx=self._conv_idx)
        return out

    def clear_cache(self):
        # Reset cache for encoder conv layers
        self._conv_idx = [0]
        self._feat_map = [None] * self._enc_conv_num
        # Reset cache for decoder conv layers
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

class WanVideoVAE(nn.Module):
    def __init__(self, z_dim=16, dim=96):
        super().__init__()
        # Mean and Std dev for normalization (hardcoded from original code)
        # These are likely normalization constants for the latent space.
        self.mean = mx.array([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921])
        self.std = mx.array([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160])
        self.scale = [self.mean, 1.0 / self.std] # Used for standardization/de-standardization
        
        self.model = VideoVAE_(z_dim=z_dim, dim=dim)
        self.upsampling_factor = 8 # Factor by which VAE upsamples/downsamples T, H, W

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        # Generates a 1D mask for feathering, useful for tiled decoding.
        x = mx.ones((length,))
        if not left_bound: x[:border_width] = mx.linspace(0, 1, border_width)
        if not right_bound: x[-border_width:] = mx.linspace(0, 1, border_width)[::-1] # Reversed linspace
        return x

    def build_mask(self, data, is_bound, border_width):
        # data: (B, T, H, W, C) - used for inferring spatial dims
        _, _, _, H, W = data.shape
        h_mask = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0]) # Mask for Height
        w_mask = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1]) # Mask for Width
        
        # Expand to 2D mask (H, W)
        mask = mx.minimum(h_mask[:, None], w_mask[None, :]) # (H, W)
        mask = mask.reshape(1, 1, 1, H, W) # (B=1, T=1, H, W, 1) for broadcasting
        return mask

    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        # hidden_states: (B, T_latent, H_latent, W_latent, C_z)
        # Decodes using tiling to manage memory for large outputs.
        
        _, T_latent, H_latent, W_latent, _ = hidden_states.shape
        size_h, size_w = tile_size # Tile size in spatial dims
        stride_h, stride_w = tile_stride # Stride for moving tiles
        
        tasks = [] # List of (h_start, h_end, w_start, w_end) for tiling
        # Iterate through spatial dimensions to define tiles
        for h in range(0, H_latent, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H_latent: continue # Avoid redundant tiles
            for w in range(0, W_latent, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W_latent: continue
                tasks.append((h, min(h + size_h, H_latent), w, min(w + size_w, W_latent)))
        
        # Output dimensions after decoding
        # VAE upsamples T by 4, H/W by 8 (assuming upsampling_factor = 8)
        out_T = T_latent * 4 - 3 # Based on original code's temporal logic
        out_H = H_latent * self.upsampling_factor
        out_W = W_latent * self.upsampling_factor
        
        # Prepare canvas to accumulate tiled results
        weight_canvas = mx.zeros((1, out_T, out_H, out_W, 1), dtype=hidden_states.dtype)
        values_canvas = mx.zeros((1, 3, out_T, out_H, out_W), dtype=hidden_states.dtype) # Output is RGB (3 channels)

        # Process each tile
        for h_start, h_end, w_start, w_end in tqdm(tasks, desc="VAE decoding tiles"):
            # Extract tile from hidden states
            tile_hs = hidden_states[:, :, h_start:h_end, w_start:w_end, :]
            
            # Decode the tile
            decoded_tile = self.model.decode(tile_hs, self.scale) # (B, T_out, H_out, W_out, 3)
            
            # Build feathering mask for blending tiles
            # Mask needs to correspond to spatial dims of the tile output
            is_bound_h = (h_start == 0, h_end >= H_latent)
            is_bound_w = (w_start == 0, w_end >= W_latent)
            # Border width for mask scaling
            border_w_h = (size_h - stride_h) * self.upsampling_factor
            border_w_w = (size_w - stride_w) * self.upsampling_factor
            mask = self.build_mask(decoded_tile, is_bound=is_bound_h + is_bound_w, border_width=(border_w_h, border_w_w))
            
            # Map decoded tile to its position on the final canvas
            target_h_start, target_w_start = h_start * self.upsampling_factor, w_start * self.upsampling_factor
            target_h_end = target_h_start + decoded_tile.shape[2]
            target_w_end = target_w_start + decoded_tile.shape[3]
            
            # Accumulate weighted results
            values_canvas[:, :, :, target_h_start:target_h_end, target_w_start:target_w_end] += (decoded_tile.transpose(0, 4, 1, 2, 3) * mask) # Transpose to (B, C, T, H, W)
            weight_canvas[:, :, target_h_start:target_h_end, target_w_start:target_w_end, :] += mask.transpose(0, 4, 1, 2, 3) # Transpose mask too
            
        # Normalize by weights to get blended output
        weight_canvas[weight_canvas == 0] = 1.0 # Avoid division by zero
        values_canvas = values_canvas / weight_canvas
        
        return values_canvas.clamp(-1.0, 1.0) # Clamp output to [-1, 1]

    def tiled_encode(self, video, device, tile_size, tile_stride):
        # video: (B, T, H, W, C_in)
        _, T, H, W, _ = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H: continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W: continue
                tasks.append((h, min(h + size_h, H), w, min(w + size_w, W)))
        
        # Output dims after encoding (downsampling)
        out_T = (T + 3) // 4 # Based on original code
        out_H = H // self.upsampling_factor
        out_W = W // self.upsampling_factor
        
        weight_canvas = mx.zeros((1, out_T, out_H, out_W, 1), dtype=video.dtype)
        values_canvas = mx.zeros((1, self.z_dim, out_T, out_H, out_W), dtype=video.dtype) # Latent space dim
        
        for h_start, h_end, w_start, w_end in tqdm(tasks, desc="VAE encoding tiles"):
            tile_video = video[:, :, h_start:h_end, w_start:w_end, :]
            
            encoded_tile = self.model.encode(tile_video, self.scale) # (B, T_enc, H_enc, W_enc, C_z)
            
            # Build mask for blending (spatial dims of encoded tile)
            is_bound_h = (h_start == 0, h_end >= H)
            is_bound_w = (w_start == 0, w_end >= W)
            border_w_h = (size_h - stride_h) // self.upsampling_factor
            border_w_w = (size_w - stride_w) // self.upsampling_factor
            mask = self.build_mask(encoded_tile, is_bound=is_bound_h + is_bound_w, border_width=(border_w_h, border_w_w))
            
            # Map encoded tile to its position on the final canvas
            target_h_start, target_w_start = h_start // self.upsampling_factor, w_start // self.upsampling_factor
            target_h_end = target_h_start + encoded_tile.shape[2]
            target_w_end = target_w_start + encoded_tile.shape[3]
            
            # Accumulate weighted results
            values_canvas[:, :, :, target_h_start:target_h_end, target_w_start:target_w_end] += (encoded_tile.transpose(0, 4, 1, 2, 3) * mask) # Transpose to (B, C_z, T, H, W)
            weight_canvas[:, :, target_h_start:target_h_end, target_w_start:target_w_end, :] += mask.transpose(0, 4, 1, 2, 3)
            
        weight_canvas[weight_canvas == 0] = 1.0
        values_canvas = values_canvas / weight_canvas
        
        return values_canvas

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        # videos: list of (T, H, W, C) tensors
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0) # Add batch dim (B=1)
            if tiled:
                # Adjust tile size for upsampling factor
                tile_size_ = (tile_size[0] * self.upsampling_factor, tile_size[1] * self.upsampling_factor)
                tile_stride_ = (tile_stride[0] * self.upsampling_factor, tile_stride[1] * self.upsampling_factor)
                hidden_state = self.tiled_encode(video, device, tile_size_, tile_stride_)
            else:
                hidden_state = self.model.encode(video, self.scale) # Single encode
            hidden_states.append(hidden_state.squeeze(0)) # Remove batch dim
        return mx.stack(hidden_states) # (N, T_latent, H_latent, W_latent, C_z)

    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        # hidden_states: (N, T_latent, H_latent, W_latent, C_z)
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0) # Add batch dim (B=1)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.model.decode(hidden_state, self.scale) # Single decode
            videos.append(video.squeeze(0)) # Remove batch dim
        return mx.stack(videos) # (N, T_out, H_out, W_out, 3)

    def clear_cache(self):
        self.model.clear_cache()

    def stream_decode(self, hidden_states, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        # hidden_states: list of (T_latent, H_latent, W_latent, C_z) tensors (one per latent slice)
        # This is for frame-by-frame decoding.
        
        # Original code expects list of length 1 for stream_decode.
        # If hidden_states is already a single tensor (B=1), reshape.
        if isinstance(hidden_states, mx.array):
            hidden_states = [hidden_states]

        # The VAE stream decode expects a single temporal slice of latents.
        # Each element in hidden_states list corresponds to one latent slice.
        # We need to pass one slice at a time to model.stream_decode.
        decoded_frames = []
        for hs_slice in hidden_states:
            hs_slice = hs_slice.unsqueeze(0) # Add batch dim (B=1)
            decoded_frame = self.model.stream_decode(hs_slice, self.scale) # (B, 1, H, W, 3)
            decoded_frames.append(decoded_frame.squeeze(0)) # Remove batch dim
        
        return mx.stack(decoded_frames) # (N_slices, T_out, H, W, 3) - where T_out is typically 1


# ==============================================================================
# LQ Projection Modules (Feature Extractors)
# ==============================================================================

class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff, self.hh, self.ww = ff, hh, ww

    def __call__(self, x):
        # Input x: (B, T, H, W, C) - Channel Last
        # Logic PyTorch: "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w"
        # Logic MLX (Channel Last): (B, T*ff, H*hh, W*ww, C) -> (B, T, H, W, C*ff*hh*ww)

        B, T_full, H_full, W_full, C = x.shape
        T, H, W = T_full // self.ff, H_full // self.hh, W_full // self.ww

        # 1. Reshape to split dimensions
        x = x.reshape(B, T, self.ff, H, self.hh, W, self.ww, C)

        # 2. Transpose to group channel-mixing dimensions at the end
        # (B, T, ff, H, hh, W, ww, C) -> (B, T, H, W, C, ff, hh, ww)
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)

        # 3. Flatten last dimensions
        x = x.reshape(B, T, H, W, C * self.ff * self.hh * self.ww)
        return x

class CausalConv3dReplicate(nn.Module):
    """
    Giống CausalConv3dZeroPad nhưng dùng 'edge' padding (tương đương replicate trong PyTorch cho biên).
    Lưu ý: MLX mx.pad hỗ trợ 'edge' (lặp lại giá trị biên).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride, stride)
        if isinstance(padding, int): padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_h = padding[1]
        self.pad_w = padding[2]
        self.pad_t_original = padding[0]
        self.causal_pad_amount = 2 * self.pad_t_original

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def __call__(self, x, cache_x=None):
        # x: (B, T, H, W, C)

        # 1. Temporal Causal Padding (Replicate/Edge mode for time?)
        # Code gốc PyTorch dùng F.pad(..., mode="replicate") cho toàn bộ padding.
        # Ở đây ta xử lý Temporal riêng.

        if self.causal_pad_amount > 0:
            if cache_x is not None:
                x = mx.concatenate([cache_x, x], axis=1)
                pad_to_add = max(0, self.causal_pad_amount - cache_x.shape[1])
                if pad_to_add > 0:
                    # Pad time dimension with 'edge' (replicate first frame)
                    # Note: mx.pad only pads with constants or edges.
                    # For causal, we strictly pad BEFORE.
                    # 'edge' padding in MLX pads both sides if specified.
                    # Construct padding tuple: ((0,0), (pre, 0), ...)
                    # However, MLX edge padding might propagate the *first* element backward.
                    x = mx.pad(x, ((0,0), (pad_to_add, 0), (0,0), (0,0), (0,0)), mode='edge')
            else:
                 x = mx.pad(x, ((0,0), (self.causal_pad_amount, 0), (0,0), (0,0), (0,0)), mode='edge')

        # 2. Spatial Padding (Replicate)
        if self.pad_h > 0 or self.pad_w > 0:
             x = mx.pad(x, ((0,0), (0,0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w), (0,0)), mode='edge')

        return self.conv(x)

class Buffer_LQ4x_Proj(nn.Module):
    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff, self.hh, self.ww = 1, 16, 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        # Input channels increased by shuffling: in_dim * 1 * 16 * 16
        c_in_shuffled = in_dim * self.ff * self.hh * self.ww

        self.conv1 = CausalConv3dReplicate(c_in_shuffled, self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.norm1 = RMSNorm(self.hidden_dim1, eps=1e-5) # images=False equivalent
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3dReplicate(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.norm2 = RMSNorm(self.hidden_dim2, eps=1e-5)
        self.act2 = nn.SiLU()

        self.linear_layers = [nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)]

        self.cache = {"conv1": None, "conv2": None}
        self.clip_idx = 0
        self.CACHE_T = 2 # Hardcoded based on globals

    def clear_cache(self):
        self.cache = {"conv1": None, "conv2": None}
        self.clip_idx = 0

    def stream_forward(self, video_clip):
        # video_clip: (B, T, H, W, C)

        if self.clip_idx == 0:
            # First frame replication logic matching PyTorch
            # video[:, :, :1, ...] -> repeat -> cat
            first_frame = video_clip[:, :1, :, :, :]
            first_frame = mx.repeat(first_frame, 3, axis=1) # Repeat time dim
            video_clip = mx.concatenate([first_frame, video_clip], axis=1)

            x = self.pixel_shuffle(video_clip)

            # Cache update
            self.cache["conv1"] = x[:, -self.CACHE_T:, :, :, :]

            # Conv1
            x = self.conv1(x, None) # No cache used for first computation block
            x = self.act1(self.norm1(x))

            # Cache update conv2
            self.cache["conv2"] = x[:, -self.CACHE_T:, :, :, :]

            self.clip_idx += 1
            return None # First clip warms up cache

        else:
            x = self.pixel_shuffle(video_clip)

            # Conv1 with cache
            cache1_x = x[:, -self.CACHE_T:, :, :, :]
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x # Update cache
            x = self.act1(self.norm1(x))

            # Conv2 with cache
            cache2_x = x[:, -self.CACHE_T:, :, :, :]
            x = self.conv2(x, self.cache["conv2"])
            self.cache["conv2"] = cache2_x
            x = self.act2(self.norm2(x))

            # Flatten spatial/temporal dims to sequence for linear projection
            # (B, T, H, W, C) -> (B, L, C)
            B, T, H, W, C = x.shape
            out_x = x.reshape(B, -1, C)

            return [layer(out_x) for layer in self.linear_layers]

class WanModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config["dim"]
        self.patch_size = config["patch_size"] # (pt, ph, pw) usually (1, 2, 2)
        self.freq_dim = config["freq_dim"]

        # 1. Embeddings
        # Patch Embedding: (B, T, H, W, C_in) -> (B, f, h, w, dim)
        self.patch_embedding = nn.Conv3d(
            config["in_dim"], self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(config["text_dim"], self.dim),
            nn.GELU(approx="tanh"),
            nn.Linear(self.dim, self.dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * 6) # 6 params per dimension for DiTBlock modulation
        )

        # 2. Transformer Blocks
        self.blocks = [
            DiTBlock(self.dim, config["num_heads"], config["ffn_dim"], config["eps"])
            for _ in range(config["num_layers"])
        ]

        # 3. Output Head
        self.head = Head(self.dim, config["out_dim"], self.patch_size, config["eps"])

        # 4. Precompute RoPE Frequencies
        head_dim = self.dim // config["num_heads"]
        # freqs_cis_3d returns (freqs_f, freqs_h, freqs_w)
        self.freqs_cis = precompute_freqs_cis_3d(head_dim)

    def reinit_cross_kv(self, context):
        """
        Khởi tạo cache Key/Value cho Cross-Attention từ embedding văn bản.
        context: (B, L_text, text_dim)
        """
        if context is None: return
        ctx_encoded = self.text_embedding(context)
        for block in self.blocks:
            block.cross_attn.init_cache(ctx_encoded)

    def patchify(self, x):
        """
        Biến đổi video latents thành chuỗi các patch tokens.
        x: (B, T, H, W, C_in)
        Returns:
            x_flat: (B, L_total, dim)
            grid_size: (f, h, w)
        """
        # MLX Conv3d expects (N, T, H, W, C) input format (Channel-Last default)
        # Assuming weights were converted/transposed correctly in converter.py

        x_patched = self.patch_embedding(x) # -> (B, f, h, w, dim)
        grid_size = x_patched.shape[1:4] # (f, h, w) - dimensions 1, 2, 3

        # Flatten spatial/temporal dims to sequence: (B, f*h*w, dim)
        x_flat = x_patched.reshape(x_patched.shape[0], -1, x_patched.shape[-1])
        return x_flat, grid_size

    def unpatchify(self, x, grid_size):
        """
        Biến đổi chuỗi tokens ngược lại thành video latents.
        x: (B, L_total, dim_out)
        grid_size: (f, h, w)
        """
        B = x.shape[0]
        f, h, w = grid_size
        pt, ph, pw = self.patch_size

        # Reshape to grid
        x = x.reshape(B, f, h, w, -1)
        C_out_channels = x.shape[-1] // (pt * ph * pw)

        # Pixel Shuffle equivalent for 3D
        # (B, f, h, w, pt*ph*pw*C) -> (B, f, h, w, pt, ph, pw, C)
        x = x.reshape(B, f, h, w, pt, ph, pw, C_out_channels)

        # Transpose to group spatial/temporal dimensions:
        # (B, f, pt, h, ph, w, pw, C) -> (B, f*pt, h*ph, w*pw, C)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)

        x = x.reshape(B, f * pt, h * ph, w * pw, C_out_channels)
        return x

    def __call__(self, x, t, LQ_latents=None, current_freqs=None):
        """
        Forward pass của DiT Backbone.

        Args:
            x: (B, T, H, W, C_in) - Latents đầu vào (noise hoặc denoised state)
            t: (B,) - Timestep tensor
            LQ_latents: List[mx.array] (Optional) - Các feature map từ video chất lượng thấp (Low Quality).
                        Mỗi phần tử trong list sẽ được cộng vào input của block DiT tương ứng (Residual Injection).
            current_freqs: mx.array (Optional) - Frequencies RoPE đã được tính sẵn cho chunk hiện tại.
                           Nếu None, sẽ tự tính toán dựa trên kích thước của x.
        """

        # 1. Time Embedding & Projection
        t_freq = sinusoidal_embedding_1d(self.freq_dim, t)
        t_emb = self.time_embedding(t_freq) # (B, dim)
        t_mod = self.time_projection(t_emb) # (B, dim*6)
        t_mod = t_mod.reshape(t_mod.shape[0], 6, self.dim) # (B, 6, dim)

        # 2. Patchify
        x, (f, h, w) = self.patchify(x) # x: (B, L_total, dim)

        # 3. RoPE Frequencies Handling
        if current_freqs is not None:
            # Streaming mode: Dùng freqs được tính toán chính xác từ Pipeline
            freqs = current_freqs
        else:
            # Standard mode: Tính freqs cho toàn bộ input
            freqs_f, freqs_h, freqs_w = self.freqs_cis

            # Slice to current dimensions
            fr_f = freqs_f[:f]
            fr_h = freqs_h[:h]
            fr_w = freqs_w[:w]

            # Broadcast to (f, h, w, D/2)
            fr_f_b = mx.broadcast_to(fr_f.reshape(f, 1, 1, -1), (f, h, w, fr_f.shape[-1]))
            fr_h_b = mx.broadcast_to(fr_h.reshape(1, h, 1, -1), (f, h, w, fr_h.shape[-1]))
            fr_w_b = mx.broadcast_to(fr_w.reshape(1, 1, w, -1), (f, h, w, fr_w.shape[-1]))

            # Concatenate & Flatten
            freqs = mx.concatenate([fr_f_b, fr_h_b, fr_w_b], axis=-1)
            freqs = freqs.reshape(-1, freqs.shape[-1]) # (L_total, D/2)

        # 4. DiT Blocks Processing
        # (Không dùng cache K/V trong model forward này vì Pipeline FlashVSR quản lý logic streaming theo chunk)
        # Tuy nhiên, nếu muốn hỗ trợ cache nội tại trong tương lai, có thể thêm vào.

        for i, block in enumerate(self.blocks):
            # --- Residual Injection (FlashVSR Core Logic) ---
            if LQ_latents is not None and i < len(LQ_latents):
                # LQ_latents[i] shape: (B, L_total, dim) - Đã được projection khớp shape
                x = x + LQ_latents[i]
            # ------------------------------------------------

            x, _, _ = block(
                x, t_mod, freqs, f, h, w,
                is_stream=False # FlashVSR xử lý chunk độc lập trong 1 step
            )

        # 5. Output Head
        # Head modulation dùng t_emb gốc (B, dim)
        x = self.head(x, t_emb)

        # 6. Unpatchify -> Video Latents
        x = self.unpatchify(x, (f, h, w))

        return x

class FlowMatchScheduler:
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003, extra_one_step=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, shift=None):
        if shift is not None: self.shift = shift
        
        # Linspace in MLX
        step_indices = mx.linspace(1.0, 0.0, num_inference_steps + 1) if self.extra_one_step else mx.linspace(1.0, 0.0, num_inference_steps)
        
        # Calculate Sigmas
        sigmas = self.sigma_min + (self.sigma_max - self.sigma_min) * step_indices
        if self.extra_one_step:
            sigmas = sigmas[:-1]
            
        # Apply Time Shift
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output, timestep_idx, sample):
        # model_output: predicted velocity (v)
        # sample: current noisy latent (x_t)
        
        sigma = self.sigmas[timestep_idx]
        
        # Determine next sigma
        if timestep_idx + 1 >= len(self.sigmas):
            sigma_next = 0.0
        else:
            sigma_next = self.sigmas[timestep_idx + 1]
            
        # Euler step: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v_t
        prev_sample = sample + model_output * (sigma_next - sigma)
        return prev_sample

# ==============================================================================
# Color Correction (Wavelet / Adain)
# ==============================================================================

def _calc_mean_std(feat, eps=1e-5):
    # feat: (B, T, H, W, C)
    # Calculate over spatial dimensions (H, W) or (T, H, W)?
    # Original PyTorch uses (N, C, -1) implying global spatial stats per channel.
    # MLX Channel Last: (B, T, H, W, C)
    var = mx.var(feat, axis=(1, 2, 3), keepdims=True) + eps
    std = mx.sqrt(var)
    mean = mx.mean(feat, axis=(1, 2, 3), keepdims=True)
    return mean, std

def _adain(content_feat, style_feat):
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean

class MLXColorCorrector(nn.Module):
    def __init__(self, levels=5):
        super().__init__()
        self.levels = levels
        # Gaussian Kernel 3x3
        self.kernel = mx.array([[0.0625, 0.125, 0.0625], 
                                [0.125, 0.25, 0.125], 
                                [0.0625, 0.125, 0.0625]]).reshape(1, 3, 3, 1)

    def _wavelet_blur(self, x, radius):
        # x: (B, H, W, C) - Treat T as B for blurring individual frames
        # Use grouped conv equivalent. MLX Conv2d supports standard input.
        # Simple separable Gaussian blur approximation or dilated conv.
        # Original uses dilated conv with groups=C.
        
        B, H, W, C = x.shape
        # Pad: (radius, radius, radius, radius)
        x_pad = mx.pad(x, ((0,0), (radius, radius), (radius, radius), (0,0)), mode='edge')
        
        # Broadcast kernel to (Out=C, H=3, W=3, In=1) for depthwise conv
        weight = mx.repeat(self.kernel, C, axis=-1) # (1, 3, 3, C) -> MLX Conv2d Weight is (Out, H, W, In) usually
        # MLX Conv2d weight: (output_channels, kernel_h, kernel_w, input_channels)
        # For depthwise (groups=C): weight should be (C, 3, 3, 1).
        
        weight = mx.repeat(self.kernel.transpose(3, 1, 2, 0), C, axis=0) # (C, 3, 3, 1)
        
        # Dilated Conv
        out = nn.conv2d(x_pad, weight, stride=1, padding=0, dilation=radius, groups=C)
        return out

    def _wavelet_decompose(self, x, levels):
        high, low = mx.zeros_like(x), x
        for i in range(levels):
            radius = 2 ** i
            blurred = self._wavelet_blur(low, radius)
            high = high + (low - blurred)
            low = blurred
        return high, low

    def __call__(self, hq_image, lq_image, clip_range=(-1.0, 1.0), method="adain"):
        # hq_image: (B, T, H, W, C)
        # Flatten T into B for processing
        B, T, H, W, C = hq_image.shape
        hq_flat = hq_image.reshape(-1, H, W, C)
        lq_flat = lq_image.reshape(-1, H, W, C)
        
        if method == "wavelet":
            c_high, _ = self._wavelet_decompose(hq_flat, self.levels)
            _, s_low = self._wavelet_decompose(lq_flat, self.levels)
            out = c_high + s_low
        elif method == "adain":
            out = _adain(hq_image, lq_image) # Use 5D version for global stats
            return mx.clip(out, clip_range[0], clip_range[1])
            
        out = mx.clip(out, clip_range[0], clip_range[1])
        return out.reshape(B, T, H, W, C)

# ==============================================================================
# Model Wrapper Function
# ==============================================================================

def model_fn_wan_video(
    dit, x, t, context, 
    LQ_latents=None, 
    t_mod=None,
    **kwargs
):
    """
    Wrapper to handle the DiT forward pass with specific arguments from the pipeline.
    """
    # 1. Add LQ Latents to input noise (Residual Connection at Input or First Block)
    # The original code adds LQ latents inside the block loop. 
    # Since we can't easily inject into the compiled WanModel.__call__, 
    # we might need to modify WanModel to accept LQ_latents, 
    # OR assume LQ_latents corresponds to x structure and add it here if possible.
    # ORIGINAL LOGIC: "x += LQ_latents[block_id]" inside the loop.
    # This implies we need to pass LQ_latents INTO WanModel.
    
    # MLX Optimization: To keep WanModel generic, we can perform the addition inside 
    # a modified call or pass it. 
    # Let's assume for this port we modify WanModel.__call__ slightly in your implementation
    # to accept `residual_latents_list` or similar.
    
    # For now, let's assume `x` is updated by model internals.
    # Calling the model:
    return dit(x, t) 
    
    # NOTE: In a rigorous port, you must modify `WanModel.__call__` to accept `LQ_latents`
    # and add `x = x + LQ_latents[i]` inside the block loop.
    # Please add `LQ_latents=None` to WanModel.__call__ arguments and the loop logic provided in Phase 2.


import numpy as np
from tqdm import tqdm
import gc

class FlashVSRPipeline:
    def __init__(self, model_manager):
        self.dit = model_manager.dit
        self.vae = model_manager.vae
        self.lq_proj = model_manager.lq_proj # Buffer_LQ4x_Proj

        # FlashVSR là distilled model, chỉ cần scheduler 1 bước giả lập
        # Chúng ta set timestep cố định là 1000.0 (noise level cao nhất) để model khử nhiễu 1 lần.
        self.timestep = mx.array([1000.0])

    # Helper function để chuẩn bị Freqs (RoPE) cho từng chunk
    def get_chunk_freqs(self, cur_process_idx, f, h, w):
        freqs_f, freqs_h, freqs_w = self.dit.freqs_cis

        if cur_process_idx == 0:
            # Chunk đầu tiên: lấy từ 0 đến f
            fr_f = freqs_f[:f]
        else:
            # Các chunk sau: trượt cửa sổ freq theo logic code gốc
            # Code gốc: dit.freqs[0][4 + cur_process_idx * 2 : 4 + cur_process_idx * 2 + f]
            start_idx = 4 + cur_process_idx * 2
            end_idx = 4 + cur_process_idx * 2 + f
            fr_f = freqs_f[start_idx : end_idx]

        fr_h = freqs_h[:h]
        fr_w = freqs_w[:w]

        # Broadcast và Combine (Logic giống WanModel)
        fr_f_b = mx.broadcast_to(fr_f.reshape(f, 1, 1, -1), (f, h, w, fr_f.shape[-1]))
        fr_h_b = mx.broadcast_to(fr_h.reshape(1, h, 1, -1), (f, h, w, fr_h.shape[-1]))
        fr_w_b = mx.broadcast_to(fr_w.reshape(1, 1, w, -1), (f, h, w, fr_w.shape[-1]))

        freqs = mx.concatenate([fr_f_b, fr_h_b, fr_w_b], axis=-1)
        freqs = freqs.reshape(-1, freqs.shape[-1])
        return freqs

    def __call__(
        self,
        LQ_video, # (B, T, H, W, C)
        seed=0,
        tiled_vae=True,
        tile_size=(34, 34), tile_stride=(18, 16), # VAE Tiling params
        color_fix=False # Đơn giản hóa để tập trung vào logic chính
    ):
        mx.random.seed(seed)
        B, T_in, H, W, C = LQ_video.shape

        # 1. Tính toán kích thước latent và padding noise
        # Logic: num_frames % 4 != 1 rounding logic handled before call usually.
        # Ở đây giả sử input đã pad đúng.
        # process_total_num logic:
        num_frames = T_in
        process_total_num = (num_frames - 1) // 8 - 2

        # Init Noise (Latents)
        # Shape: (1, T_latent, H_latent, W_latent, 16)
        # Code gốc: noise = generate_noise((1, 16, (num_frames - 1) // 4 + 1, ...))
        # MLX: (1, (num_frames - 1) // 4 + 1, ..., 16)
        latent_T = (num_frames - 1) // 4 + 1
        latent_H = H // 8
        latent_W = W // 8

        latents = mx.random.normal((1, latent_T, latent_H, latent_W, 16))

        print(f"[FlashVSR] Process Total Num: {process_total_num}")
        self.lq_proj.clear_cache()
        latents_output_total = []

        LQ_pre_idx = 0
        LQ_cur_idx = 0

        # 2. Main Streaming Loop
        for cur_process_idx in tqdm(range(process_total_num)):

            # --- A. LQ Feature Extraction (Precise Mapping) ---
            LQ_latents = None # List of tensors for current chunk

            if cur_process_idx == 0:
                inner_loop_num = 7
                # Loop này mô phỏng việc stream qua từng sub-chunk nhỏ để build feature cho chunk lớn đầu tiên
                for inner_idx in range(inner_loop_num):
                    # Logic Slice: max(0, inner_idx * 4 - 3) : (inner_idx + 1) * 4 - 3
                    start = max(0, inner_idx * 4 - 3)
                    end = (inner_idx + 1) * 4 - 3
                    # Slice video (MLX slice ok với index âm/vượt quá, nó tự clamp như Python list, nhưng cần cẩn thận)
                    # Cần đảm bảo index > 0 cho MLX nếu slice không hỗ trợ logic như PyTorch
                    if start < 0: start = 0 # Safety
                    if end < 0: end = 0

                    clip = LQ_video[:, start:end, :, :, :]

                    # Stream forward
                    cur = self.lq_proj.stream_forward(clip)

                    if cur is None: continue # Skip warmup frame

                    # Accumulate features (Concatenate along Sequence/Time dimension)
                    # cur là list các tensor [layer1_feat, layer2_feat, ...]
                    if LQ_latents is None:
                        LQ_latents = cur
                    else:
                        for layer_idx in range(len(LQ_latents)):
                            # Concatenate along axis=1 (Sequence length dimension in MLX output of lq_proj)
                            # lq_proj output format: (B, L, C)
                            LQ_latents[layer_idx] = mx.concatenate([LQ_latents[layer_idx], cur[layer_idx]], axis=1)

                # Logic xác định vị trí Latent tương ứng
                # cur_latents = latents[:, :, :6, :, :]
                cur_latents_input = latents[:, :6, :, :, :] # MLX: T is dim 1
                LQ_cur_idx = (inner_loop_num - 1) * 4 - 3

            else:
                inner_loop_num = 2
                for inner_idx in range(inner_loop_num):
                    # Logic Slice: cur_process_idx * 8 + 17 + inner_idx * 4 ...
                    base = cur_process_idx * 8 + 17
                    start = base + inner_idx * 4
                    end = base + 4 + inner_idx * 4

                    clip = LQ_video[:, start:end, :, :, :]
                    cur = self.lq_proj.stream_forward(clip)

                    if cur is None: continue

                    if LQ_latents is None:
                        LQ_latents = cur
                    else:
                        for layer_idx in range(len(LQ_latents)):
                            LQ_latents[layer_idx] = mx.concatenate([LQ_latents[layer_idx], cur[layer_idx]], axis=1)

                # Logic xác định vị trí Latent
                # cur_latents = latents[:, :, 4 + cur_process_idx * 2 : 6 + cur_process_idx * 2, :, :]
                start_lat = 4 + cur_process_idx * 2
                end_lat = 6 + cur_process_idx * 2
                cur_latents_input = latents[:, start_lat:end_lat, :, :, :]

                LQ_cur_idx = cur_process_idx * 8 + 21 + (inner_loop_num - 2) * 4

            # --- B. DiT Forward Pass (1-Step Inference) ---

            # Tính toán patch grid size (f, h, w) để chuẩn bị Freqs
            # WanModel.patchify sẽ trả về (f, h, w). Ta có thể giả lập tính toán hoặc gọi patchify.
            # Để tối ưu, ta gọi patchify nhẹ hoặc tính tay:
            # f = cur_latents_input.shape[1] (T) -> sau patch embedding
            # Patch size của WanModel là (1, 2, 2).
            # Vậy f = T_chunk // 1 = T_chunk. h = H // 2. w = W // 2.
            # cur_latents_input đang là Latent space (đã downsample H/W bởi 8).
            # WanModel patchify chia tiếp H/W cho 2.
            # T_chunk ở đây là 6 (chunk đầu) hoặc 2 (chunk sau).

            f_chunk = cur_latents_input.shape[1] # T dimension
            h_chunk = cur_latents_input.shape[2] // 2
            w_chunk = cur_latents_input.shape[3] // 2

            # Lấy Freqs chính xác cho chunk này
            current_freqs = self.get_chunk_freqs(cur_process_idx, f_chunk, h_chunk, w_chunk)

            # Gọi Model
            # Lưu ý: FlashVSR là distilled model (Turbo), output của model chính là `noise_pred`.
            # Ta thực hiện 1 bước trừ noise trực tiếp: latent - model(latent)

            noise_pred = self.dit(
                cur_latents_input,
                self.timestep,
                LQ_latents=LQ_latents,
                current_freqs=current_freqs
            )

            # Apply 1-step Denoising
            # Code gốc: cur_latents = cur_latents - noise_pred_posi
            denoised_chunk = cur_latents_input - noise_pred

            latents_output_total.append(denoised_chunk)
            LQ_pre_idx = LQ_cur_idx

            # Quan trọng: Dọn dẹp bộ nhớ đồ thị MLX sau mỗi chunk
            mx.eval(denoised_chunk)
            del LQ_latents, noise_pred, current_freqs

        # 3. Concatenate & Decode
        self.lq_proj.clear_cache()

        # Nối các latent đã khử nhiễu lại theo chiều thời gian
        # Code gốc: latents = torch.cat(latents_total, dim=2) (dim 2 là T trong PyTorch B,C,T,H,W)
        # MLX: dim 1 là T (B, T, H, W, C)
        final_latents = mx.concatenate(latents_output_total, axis=1)

        print("[FlashVSR] Decoding VAE...")
        # Cần transpose latent để khớp với đầu vào VAE (B, T, H, W, C) - VAE code của chúng ta đã hỗ trợ

        # Lưu ý: Cần dùng `LQ_video` làm điều kiện (cond) cho VAE decode như code gốc
        # Code gốc: frames = self.TCDecoder.decode_video(..., cond=LQ_video[:, :, :LQ_cur_idx, :, :])
        # Phần này nằm ở VAE implementation (TCDecoder / TAEHV).
        # Vì porting TAEHV phức tạp, chúng ta dùng WanVideoVAE tiêu chuẩn.
        # Nếu muốn chất lượng cao nhất, cần port `TAEHV` (Temporal Autoencoder).
        # Ở đây tôi dùng WanVideoVAE đã port ở Phase 3.

        if tiled_vae:
             decoded = self.vae.decode([final_latents], device=None, tiled=True, tile_size=tile_size, tile_stride=tile_stride)
        else:
             decoded = self.vae.decode([final_latents], device=None, tiled=False)

        decoded = decoded.squeeze(0) # Remove batch

        # Post-process
        decoded = (decoded + 1.0) / 2.0
        decoded = mx.clip(decoded, 0.0, 1.0)

        return decoded
