import torch
from typing import List, Tuple
import gc
import numpy as np

# ComfyUI Imports
import comfy.samplers
import node_helpers
import comfy.clip_vision
import comfy.utils
import comfy.model_management
import comfy.model_sampling
import nodes

try:
    from color_matcher import ColorMatcher
    COLOR_MATCHER_AVAILABLE = True
except ImportError:
    COLOR_MATCHER_AVAILABLE = False

# ====================================================================================================
# Logging Utility
# ====================================================================================================
def _log(msg: str):
    """Simple print logger for the console."""
    print(f"[WanVideoLooper] {msg}", flush=True)

if not COLOR_MATCHER_AVAILABLE:
    _log("#####################################################################")
    _log("Warning: 'color-matcher' library not found.")
    _log("Internal color matching will be disabled.")
    _log("To enable, please install the library:")
    _log("pip install color-matcher")
    _log("#####################################################################")


# ====================================================================================================
# Core Tensor Utilities
# ====================================================================================================
def _align_to_multiple(x, multiple=16): return max(multiple, (int(x) // multiple) * multiple)
def _get_model_device_info(model):
    try:
        model_device = next(model.model.parameters()).device
        model_dtype = next(model.model.parameters()).dtype
    except Exception:
        model_device = comfy.model_management.intermediate_device()
        model_dtype = torch.float32
    return model_device, model_dtype
def _move_tensor(tensor: torch.Tensor, device, dtype):
    if tensor.device != device or tensor.dtype != dtype: return tensor.to(device=device, dtype=dtype, non_blocking=True)
    return tensor

# ====================================================================================================
# Model & Sampling Helpers (MoE Logic)
# ====================================================================================================
def _apply_sigma_shift(model, sigma_shift):
    model_sampler = model.get_model_object("model_sampling")
    if not model_sampler:
        class CustomModelSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST): pass
        model_sampler = CustomModelSampling(model.model.model_config)
    model_sampler.set_parameters(shift=sigma_shift, multiplier=1000); model.add_object_patch("model_sampling", model_sampler)
    return model
def _find_model_switch_index(model, scheduler_name, total_steps, boundary_point):
    try:
        model_sampler = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampler, scheduler_name, total_steps)
        timesteps = [model_sampler.timestep(sigma) / 1000.0 for sigma in sigmas.tolist()]
        for i, t in enumerate(timesteps):
            if t < boundary_point: return max(0, i - 1)
        return total_steps
    except Exception as e: _log(f"Warning: Error calculating model switch point, defaulting 50/50. {e}"); return total_steps // 2
def _run_sampler_step(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, disable_noise=False, start_step=None, last_step=None,  force_full_denoise=False):
    latent_dict = {"samples": latent["samples"]}
    output, = nodes.common_ksampler(model=model, seed=int(seed), steps=int(steps), cfg=float(cfg), sampler_name=str(sampler_name), scheduler=str(scheduler), positive=positive, negative=negative, latent=latent_dict, denoise=float(denoise), disable_noise=disable_noise, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise); return {"samples": output["samples"]}

# ====================================================================================================
# Conditioning & Latent Helpers
# ====================================================================================================
def _encode_seed_image_vision(clip_vision, seed_image, cropping="center"):
    if clip_vision is None: return None
    if not isinstance(seed_image, torch.Tensor): return None
    image_tensor = seed_image.to(torch.float32).clamp(0, 1)
    if image_tensor.dim() == 3: image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.dim() == 4 and image_tensor.shape[1] in (1, 3, 4): image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous()
    if image_tensor.dim() != 4 or image_tensor.shape[-1] not in (1, 3, 4): _log(f"Warning: Invalid image shape for CLIP Vision: {tuple(image_tensor.shape)}"); return None
    vision_output = clip_vision.encode_image(image_tensor, crop=(cropping == "center")); return vision_output
def _tokenize_and_encode(clip, positive_prompt, negative_prompt):
    pos_tokens = clip.tokenize(positive_prompt or "")
    pos_data = clip.encode_from_tokens(pos_tokens, return_pooled=True, return_dict=True)
    positive_cond = [(pos_data["cond"], {"pooled_output": pos_data.get("pooled_output")})]
    neg_tokens = clip.tokenize(negative_prompt or "")
    neg_data = clip.encode_from_tokens(neg_tokens, return_pooled=True, return_dict=True)
    negative_cond = [(neg_data["cond"], {"pooled_output": neg_data.get("pooled_output")})]
    return positive_cond, negative_cond
def _prepare_latent_window(positive_cond, negative_cond, vae, width, height, frame_count, batch_size, seed_image=None, vision_output=None, pre_allocated_latent=None):
    device = comfy.model_management.intermediate_device()
    latent_height = height // 8; latent_width = width // 8; time_dim = (frame_count + 3) // 4
    target_shape = [batch_size, 16, time_dim, latent_height, latent_width]
    
    if pre_allocated_latent is not None and pre_allocated_latent.shape == torch.Size(target_shape) and pre_allocated_latent.device == device:
        empty_latent = pre_allocated_latent.zero_()
    else:
        empty_latent = torch.zeros(target_shape, device=device)
    if seed_image is not None:
        seed_image_upscaled = comfy.utils.common_upscale(seed_image[:frame_count].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        image_padded = torch.ones((frame_count, height, width, seed_image_upscaled.shape[-1]), device=seed_image_upscaled.device, dtype=seed_image_upscaled.dtype) * 0.5
        image_padded[:seed_image_upscaled.shape[0]] = seed_image_upscaled
        seed_image_latent = vae.encode(image_padded[:, :, :, :3])
        mask_time_dim = int(seed_image_latent.size(2))
        mask = torch.ones((1, 1, mask_time_dim, latent_height, latent_width), device=seed_image.device, dtype=seed_image.dtype)
        zero_frames_dim = min(mask_time_dim, ((seed_image_upscaled.shape[0] - 1) // 4) + 1)
        mask[:, :, :zero_frames_dim] = 0.0
        positive_cond = node_helpers.conditioning_set_values(positive_cond, {"concat_latent_image": seed_image_latent, "concat_mask": mask})
        negative_cond = node_helpers.conditioning_set_values(negative_cond, {"concat_latent_image": seed_image_latent, "concat_mask": mask})
    if vision_output is not None:
        positive_cond = node_helpers.conditioning_set_values(positive_cond, {"clip_vision_output": vision_output})
        negative_cond = node_helpers.conditioning_set_values(negative_cond, {"clip_vision_output": vision_output})
    return positive_cond, negative_cond, {"samples": empty_latent}

# ====================================================================================================
# Color Match Helper (GPU Accelerated)
# ====================================================================================================
def _calculate_covariance(x):
    """Calculate the covariance matrix of a tensor."""
    x_mean = x.mean(dim=0)
    x_centered = x - x_mean
    return torch.matmul(x_centered.T, x_centered) / (x.shape[0] - 1)

def _rgb_to_xyz(rgb):
    """Converts a PyTorch tensor from RGB to XYZ color space."""
    M = torch.tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ], device=rgb.device, dtype=rgb.dtype)
    
    if rgb.dim() == 3:
        h, w = rgb.shape[1:]
        xyz = torch.matmul(M, rgb.view(3, -1)).view(3, h, w)
    elif rgb.dim() == 4:
        b, h, w = rgb.shape[0], rgb.shape[2], rgb.shape[3]
        xyz = torch.matmul(M, rgb.view(b, 3, -1)).view(b, 3, h, w)
    else:
        raise ValueError("Input RGB tensor must be 3D (C, H, W) or 4D (B, C, H, W)")
    return xyz

def _xyz_to_lab(xyz):
    """Converts a PyTorch tensor from XYZ to Lab color space."""
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    xyz_norm = torch.empty_like(xyz)
    xyz_norm[:, 0] = xyz[:, 0] / Xn
    xyz_norm[:, 1] = xyz[:, 1] / Yn
    xyz_norm[:, 2] = xyz[:, 2] / Zn

    delta = 6/29
    f_t = torch.where(xyz_norm > delta**3, xyz_norm**(1/3), (xyz_norm / (3 * delta**2)) + (4/29))

    L = (116 * f_t[:, 1]) - 16
    a = 500 * (f_t[:, 0] - f_t[:, 1])
    b = 200 * (f_t[:, 1] - f_t[:, 2])
    return torch.stack([L, a, b], dim=1)

def _lab_to_xyz(lab):
    """Converts a PyTorch tensor from Lab to XYZ color space."""
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    f_y = (lab[:, 0] + 16) / 116
    f_x = lab[:, 1] / 500 + f_y
    f_z = f_y - lab[:, 2] / 200

    delta = 6/29
    x_norm = torch.where(f_x > delta, f_x**3, (f_x - 16/116) / 7.787)
    y_norm = torch.where(f_y > delta, f_y**3, (f_y - 16/116) / 7.787)
    z_norm = torch.where(f_z > delta, f_z**3, (f_z - 16/116) / 7.787)

    X = x_norm * Xn
    Y = y_norm * Yn
    Z = z_norm * Zn
    return torch.stack([X, Y, Z], dim=1)

def _xyz_to_rgb(xyz):
    """Converts a PyTorch tensor from XYZ to RGB color space."""
    M_inv = torch.tensor([
        [ 3.240479, -1.537150, -0.498535],
        [-0.969256,  1.875992,  0.041556],
        [ 0.055648, -0.204043,  1.057311]
    ], device=xyz.device, dtype=xyz.dtype)

    if xyz.dim() == 3:
        h, w = xyz.shape[1:]
        rgb = torch.matmul(M_inv, xyz.view(3, -1)).view(3, h, w)
    elif xyz.dim() == 4:
        b, h, w = xyz.shape[0], xyz.shape[2], xyz.shape[3]
        rgb = torch.matmul(M_inv, xyz.view(b, 3, -1)).view(b, 3, h, w)
    else:
        raise ValueError("Input XYZ tensor must be 3D (C, H, W) or 4D (B, C, H, W)")
    return rgb

def _rgb_to_lab_torch(rgb_image):
    """Converts an RGB image tensor to Lab color space."""
    xyz_image = _rgb_to_xyz(rgb_image)
    lab_image = _xyz_to_lab(xyz_image)
    return lab_image

def _lab_to_rgb_torch(lab_image):
    """Converts a Lab image tensor to RGB color space."""
    xyz_image = _lab_to_xyz(lab_image)
    rgb_image = _xyz_to_rgb(xyz_image)
    return rgb_image

def _reinhard_color_transfer(source_image_rgb, target_image_rgb):
    """Performs Reinhard color transfer from target_image_rgb to source_image_rgb."""
    if source_image_rgb.dim() == 3:
        source_image_rgb = source_image_rgb.unsqueeze(0)
    if target_image_rgb.dim() == 3:
        target_image_rgb = target_image_rgb.unsqueeze(0)

    source_lab = _rgb_to_lab_torch(source_image_rgb)
    target_lab = _rgb_to_lab_torch(target_image_rgb)

    mean_source = torch.mean(source_lab, dim=[2, 3], keepdim=True)
    std_source = torch.std(source_lab, dim=[2, 3], keepdim=True)
    mean_target = torch.mean(target_lab, dim=[2, 3], keepdim=True)
    std_target = torch.std(target_lab, dim=[2, 3], keepdim=True)

    transferred_lab = (source_lab - mean_source) * (std_target / (std_source + 1e-8)) + mean_target
    transferred_rgb = _lab_to_rgb_torch(transferred_lab)
    transferred_rgb = torch.clamp(transferred_rgb, 0, 1)

    if source_image_rgb.shape[0] == 1:
        return transferred_rgb.squeeze(0)
    return transferred_rgb

def _gpu_mkl(cov_source, cov_target):
    """PyTorch implementation of the MKL algorithm."""
    EPS = 1e-8
    eigvals_source, eigvecs_source = torch.linalg.eigh(cov_source)
    eigvals_source = eigvals_source.real
    eigvecs_source = eigvecs_source.real

    sqrt_eigvals_source = torch.sqrt(torch.clamp(eigvals_source, min=0)) + EPS

    C = torch.matmul(torch.matmul(torch.diag(sqrt_eigvals_source), eigvecs_source.T), torch.matmul(cov_target, torch.matmul(eigvecs_source, torch.diag(sqrt_eigvals_source))))

    eigvals_C, eigvecs_C = torch.linalg.eigh(C)
    eigvals_C = eigvals_C.real
    eigvecs_C = eigvecs_C.real

    sqrt_eigvals_C = torch.sqrt(torch.clamp(eigvals_C, min=0)) + EPS
    inv_sqrt_eigvals_source = torch.diag(1.0 / sqrt_eigvals_source)

    transform_matrix = torch.matmul(torch.matmul(torch.matmul(torch.matmul(eigvecs_source, inv_sqrt_eigvals_source), eigvecs_C), torch.diag(sqrt_eigvals_C)), torch.matmul(eigvecs_C.T, torch.matmul(inv_sqrt_eigvals_source, eigvecs_source.T)))
    return transform_matrix

def _gpu_mkl_color_transfer(source, target):
    """PyTorch implementation of the MKL color transfer."""
    b, c, h, w = source.shape
    source_reshaped = source.permute(0, 2, 3, 1).reshape(-1, 3)
    target_reshaped = target.permute(0, 2, 3, 1).reshape(-1, 3)

    cov_source = _calculate_covariance(source_reshaped)
    cov_target = _calculate_covariance(target_reshaped)

    transform_matrix = _gpu_mkl(cov_source, cov_target)

    mean_source = source_reshaped.mean(dim=0)
    mean_target = target_reshaped.mean(dim=0)

    result = torch.matmul(source_reshaped - mean_source, transform_matrix) + mean_target
    result = result.reshape(b, h, w, c).permute(0, 3, 1, 2)

    return torch.clamp(result, 0, 1)

# ====================================================================================================
# Color Match Helper
# ====================================================================================================
def _apply_color_match(target_batch_tensor, reference_tensor, strength=1.0, method="Reinhard"):
    """Applies color matching from reference to target batch using GPU."""
    if not COLOR_MATCHER_AVAILABLE:
        _log("Skipping color match: color-matcher library not available.")
        return target_batch_tensor

    if reference_tensor is None or target_batch_tensor is None:
        return target_batch_tensor

    if strength <= 0:
        _log("Skipping color match: strength is 0 or less.")
        return target_batch_tensor

    _log(f"Applying internal color match (strength: {strength:.2f}) using {method} method on GPU.")

    original_device = target_batch_tensor.device
    gpu_device = comfy.model_management.intermediate_device()

    # Move tensors to GPU for processing
    target_gpu = target_batch_tensor.to(gpu_device)
    ref_gpu = reference_tensor.to(gpu_device)

    # Ensure tensors are in (B, C, H, W) format
    if target_gpu.dim() == 4 and target_gpu.shape[-1] == 3:
        target_gpu = target_gpu.permute(0, 3, 1, 2)
    if ref_gpu.dim() == 4 and ref_gpu.shape[-1] == 3:
        ref_gpu = ref_gpu.permute(0, 3, 1, 2)

    if method == "MKL":
        matched_batch = _gpu_mkl_color_transfer(target_gpu, ref_gpu)
    else:  # Default to Reinhard
        matched_batch = _reinhard_color_transfer(target_gpu, ref_gpu)

    # Blend the original and color-matched images on the GPU
    final_result = target_gpu + strength * (matched_batch - target_gpu)
    final_result = torch.clamp(final_result, 0.0, 1.0)

    # Permute back to (B, H, W, C)
    if final_result.dim() == 4 and final_result.shape[1] == 3:
        final_result = final_result.permute(0, 2, 3, 1)

    return final_result.to(original_device)


# ====================================================================================================
# Main WanVideoLooper Node Class
# ====================================================================================================
class WanVideoLooper:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL", {"tooltip": "The BASE high-noise model (can be pre-patched with global LoRAs)."}),
                "model_low": ("MODEL", {"tooltip": "The BASE low-noise model (can be pre-patched with global LoRAs)."}),
                "clip": ("CLIP", {"tooltip": "The BASE CLIP model (can be pre-patched with global LoRAs)."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the final image."}),
                "start_image": ("IMAGE", {"tooltip": "The seed image that the first loop will start from."}),
                "positive_prompts": ("*", {"forceInput": True, "tooltip": "Connect the 'prompt_list' output from the WanVideo Multi-Prompt node."}),
                "negative_prompt": ("STRING", {"tooltip": "A single, global negative prompt applied to all loops."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed. This will be used for all loops."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "The total number of sampling steps for each loop."}),
                "cfg_motion_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "CFG for the first step ONLY. A value > 1.0 will engage the negative prompt for initial motion guidance. Set to 1.0 to disable motion guidance."}),
                "cfg_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-Free Guidance scale for both high and low noise models. 1.0 disables guidance."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampling algorithm (e.g., dpmpp_2m_sde_gpu)."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The noise scheduler (e.g., karras)."}),
                "model_switch_point": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001, "tooltip": "Timestep to switch from high-noise to low-noise model. 0.9 (I2V) or 0.875 (T2V) recommended."}),
                "sigma_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.01, "tooltip": "Applies sigma shift to both models. 8.0 is a good default."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise amount. 1.0 means generate a full new image."}),

                "width": ("INT", {"default": 832, "min": 64, "max": 8192, "step": 16, "tooltip": "The width of the output video."}),
                "height": ("INT", {"default": 480, "min": 64, "max": 8192, "step": 16, "tooltip": "The height of the output video."}),
                "frame_merge": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1, "tooltip": "Number of frames to overlap/merge between loops for smooth transitions."}),
                "duration_sec": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1, "tooltip": "Duration of each segment in seconds. 5s is recommended, max 10."}),
                "color_match_method": (["Disabled", "Reinhard (GPU)", "MKL (GPU)"], {"default": "Disabled", "tooltip": "'Disabled': No color matching is applied. 'Reinhard (GPU)': Applies Reinhard color transfer. 'MKL (GPU)': Applies MKL color transfer."}),
                "color_match_reference_frame": (["First Frame", "Last Frame", "Sequential", "Reference"], {"default": "First Frame", "tooltip": "'First Frame': Matches all frames to the very first frame of the entire sequence. 'Last Frame': Matches all frames to the very last frame of the entire sequence. 'Sequential': Matches each segment to the last frame of the previous segment. 'Reference': Matches all frames to the optional 'color_match_ref' image."}),

                "color_match_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of the internal color matching (1.0 = full match)."}),
                "enable_dry_run": ("BOOLEAN", {"default": False, "tooltip": "If enabled, skips sampling and decoding to quickly check setup and logs."}),


            },
            "optional": {
                "clip_vision": ("CLIP_VISION", {"tooltip": "(Optional) A CLIP Vision model for guiding the start image."}),
                "model_clip_sequence": ("ANY", {"forceInput": True, "tooltip": "(Optional) Connect a WanVideo Lora Sequencer to use different pre-patched models/clips per segment."}),
                "color_match_ref": ("IMAGE", {"tooltip": "(Optional) Static reference image for internal color matching. If omitted, uses last frame of previous segment."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "last_frame")
    FUNCTION = "loop_video"
    CATEGORY = "WanVideoLooper"

    # --- Main Looping Function ---
    def loop_video(
        self,
        model_high, model_low, clip, vae,
        start_image, positive_prompts, negative_prompt,
        seed, steps, cfg_motion_noise, cfg_noise,
        sampler_name, scheduler, model_switch_point, sigma_shift, denoise,
        width, height, frame_merge, duration_sec, enable_dry_run,
        color_match_strength, color_match_method, color_match_reference_frame,
        clip_vision=None,
        model_clip_sequence=None,
        color_match_ref=None 
    ):

        # --- 1. Validation & Initial Setup ---
        if not isinstance(positive_prompts, list):
            _log("Error: 'positive_prompts' is not a valid list..."); return (torch.zeros((1, 64, 64, 3)),)*2

        valid_sequence = isinstance(model_clip_sequence, list)
        if not valid_sequence and model_clip_sequence is not None:
             _log("Warning: Connected 'model_clip_sequence' is not a valid list (likely bypassed). Ignoring.");
             model_clip_sequence = None

        all_frames_collected = []
        current_seed_image = start_image
        merge_frame_count = int(frame_merge)
        duration = max(1, int(duration_sec))
        window_frame_count = (duration * 16) + 1

        valid_sequence = isinstance(model_clip_sequence, list)

        if valid_sequence:
            _log("Using input models directly as base for sequence processing (no initial cloning).")
            base_model_high = model_high
            base_model_low = model_low
            base_clip = clip
        else:
            _log("Cloning base models and clip for looping (no sequence provided).")
            base_model_high = model_high.clone()
            base_model_low = model_low.clone()
            base_clip = clip.clone()

        _log(f"Starting loop with {len(positive_prompts)} prompts...")
        if enable_dry_run: _log("!!! DRY RUN MODE ENABLED - Sampling and decoding will be skipped !!!")
        if color_match_method != "Disabled" and not COLOR_MATCHER_AVAILABLE:
             _log("Warning: Internal color match enabled, but 'color-matcher' library is missing. Feature disabled.")
             color_match_method = "Disabled"
        if color_match_method != "Disabled" and color_match_reference_frame in ["Last Frame", "First Frame", "Reference"]:
             _log(f"'{color_match_reference_frame}' color match is enabled. Segment-by-segment matching will be skipped.")
        elif color_match_method != "Disabled" and color_match_reference_frame == "Sequential":
             _log(f"'Sequential' color match is enabled. Each segment will be matched to the previous segment's last frame.")


        model_device, model_dtype = _get_model_device_info(base_model_high)
        cpu_device = torch.device("cpu")
        render_width = _align_to_multiple(width, 16)
        render_height = _align_to_multiple(height, 16)
        current_seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        total_steps = int(steps)
        _log("Pre-allocating latent tensor for reuse.")
        # Calculate target shape for pre-allocated latent
        latent_height_for_prealloc = render_height // 8
        latent_width_for_prealloc = render_width // 8
        time_dim_for_prealloc = (window_frame_count + 3) // 4
        pre_allocated_latent_tensor = torch.zeros([1, 16, time_dim_for_prealloc, latent_height_for_prealloc, latent_width_for_prealloc], device=comfy.model_management.intermediate_device())

        # --- Color Match Setup ---
        static_color_reference = color_match_ref[0:1] if color_match_ref is not None else None
        previous_loop_last_frame = None
        first_frame_for_color_match = None

        # --- 2. Pre-Loop Preparations (Applied Once) ---
        _log(f"Applying Sigma Shift: {sigma_shift}")
        base_model_high = _apply_sigma_shift(base_model_high, sigma_shift)
        base_model_low = _apply_sigma_shift(base_model_low, sigma_shift)
        _, global_negative_cond = _tokenize_and_encode(base_clip, "", negative_prompt)
        switch_step_index = _find_model_switch_index(base_model_high, scheduler, total_steps, model_switch_point)
        _log(f"Model switch point {model_switch_point} calculated split at step: {switch_step_index} / {total_steps}")
        if switch_step_index >= total_steps: switch_step_index = total_steps
        elif switch_step_index <= 0: switch_step_index = 0

        # --- 3. Main Generation Loop (Per Prompt) ---
        for i, prompt_text in enumerate(positive_prompts):
            _log(f"--- Starting Loop {i+1}/{len(positive_prompts)} ---")
            _log(f"Prompt: {prompt_text[:80]}...")

            # --- 3a. Select/Clone Models for this Loop ---
            if not valid_sequence:
                _log(f"Segment {i+1}: Using BASE models and clip (no sequence provided).")
                loop_model_high = base_model_high
                loop_model_low = base_model_low
                loop_clip = base_clip
                segment_models_provided = False
                using_provided_high, using_provided_low, using_provided_clip = False, False, False
            else:
                _log(f"Segment {i+1}: Processing models and clip from sequence.")
                loop_model_high = base_model_high # Start with base, will be overwritten if sequence provides
                loop_model_low = base_model_low   # Start with base, will be overwritten if sequence provides
                loop_clip = base_clip
                segment_models_provided = False
                using_provided_high, using_provided_low, using_provided_clip = False, False, False
                if i < len(model_clip_sequence):
                    segment_data = model_clip_sequence[i]
                    if segment_data and isinstance(segment_data, tuple) and len(segment_data) == 3:
                        model_high_seq, model_low_seq, clip_seq = segment_data
                        if model_high_seq is not None: loop_model_high = model_high_seq; segment_models_provided = True; using_provided_high = True
                        if model_low_seq is not None: loop_model_low = model_low_seq; segment_models_provided = True; using_provided_low = True
                        if clip_seq is not None: loop_clip = clip_seq; segment_models_provided = True; using_provided_clip = True

            if segment_models_provided:
                high_source = "PROVIDED (LoRA?)" if using_provided_high else "CLONED BASE"
                low_source = "PROVIDED (LoRA?)" if using_provided_low else "CLONED BASE"
                clip_source = "PROVIDED (LoRA?)" if using_provided_clip else "CLONED BASE"
                _log(f"Segment {i+1}: Using models - High: {high_source}, Low: {low_source}, Clip: {clip_source}")
            else:
                 _log(f"Segment {i+1}: Using CLONED BASE models and clip (no specific models provided for this segment).")

            # --- 3b. Prepare Conditionings and Latent ---
            global_positive_cond, _ = _tokenize_and_encode(loop_clip, prompt_text, "")
            vision_output = _encode_seed_image_vision(clip_vision, current_seed_image, "center")
            loop_positive_cond, loop_negative_cond, latent_on_cpu = _prepare_latent_window(
                global_positive_cond, global_negative_cond, vae,
                render_width, render_height, window_frame_count, 1,
                current_seed_image, vision_output, pre_allocated_latent=pre_allocated_latent_tensor
            )
            latent_on_gpu = {"samples": _move_tensor(latent_on_cpu["samples"], model_device, model_dtype)}
            final_latent = None

            # --- 3c/3d. Run Samplers OR Skip for Dry Run ---
            if not enable_dry_run:
                _log(f"Segment {i+1}: Running HIGH noise sampler (Steps 0 to {switch_step_index}).")
                if cfg_motion_noise != 1.0 and switch_step_index > 1:
                    _log(f"Segment {i+1}: Using special Motion CFG ({cfg_motion_noise}) for step 0.")
                    initial_latent = _run_sampler_step(loop_model_high, current_seed, total_steps, cfg_motion_noise, sampler_name, scheduler, loop_positive_cond, loop_negative_cond, latent_on_gpu, denoise, disable_noise=False, start_step=0, last_step=1, force_full_denoise=False)
                    high_noise_latent = _run_sampler_step(loop_model_high, current_seed, total_steps, cfg_noise, sampler_name, scheduler, loop_positive_cond, loop_negative_cond, initial_latent, denoise, disable_noise=True, start_step=1, last_step=switch_step_index, force_full_denoise=False)
                else:
                    high_noise_latent = _run_sampler_step(loop_model_high, current_seed, total_steps, cfg_noise, sampler_name, scheduler, loop_positive_cond, loop_negative_cond, latent_on_gpu, denoise, disable_noise=False, start_step=0, last_step=switch_step_index, force_full_denoise=False)

                _log(f"Segment {i+1}: Running LOW noise sampler (Steps {switch_step_index} to {total_steps}).")
                final_latent = _run_sampler_step(loop_model_low, current_seed, total_steps, cfg_noise, sampler_name, scheduler, loop_positive_cond, loop_negative_cond, high_noise_latent, denoise, disable_noise=True, start_step=switch_step_index, last_step=total_steps, force_full_denoise=True)
                _log(f"Segment {i+1}: Sampling finished.")
            else:
                _log(f"Segment {i+1}: DRY RUN - Skipping HIGH noise sampler."); _log(f"Segment {i+1}: DRY RUN - Skipping LOW noise sampler.")
                final_latent = latent_on_gpu; high_noise_latent = latent_on_gpu

            # --- 3e. Decode & Normalize OR Skip for Dry Run ---
            decoded_image_batch = None
            if not enable_dry_run and final_latent is not None:
                _log(f"Segment {i+1}: Decoding latent.")
                final_latent_on_cpu = _move_tensor(final_latent["samples"], cpu_device, torch.float32)
                decoded_data = vae.decode(final_latent_on_cpu)
                decoded_image_batch = decoded_data["image"] if isinstance(decoded_data, dict) else decoded_data
                if decoded_image_batch.dim() == 3: decoded_image_batch = decoded_image_batch.unsqueeze(0)
                elif decoded_image_batch.dim() == 5:
                    if decoded_image_batch.shape[-1] not in (1, 3, 4): decoded_image_batch = decoded_image_batch.permute(0, 2, 3, 4, 1).contiguous()
                    b, t, h, w, c = decoded_image_batch.shape; decoded_image_batch = decoded_image_batch.reshape(b * t, h, w, c)
                elif decoded_image_batch.dim() == 4:
                    if decoded_image_batch.shape[1] in (1, 3, 4) and decoded_image_batch.shape[-1] not in (1, 3, 4):
                        decoded_image_batch = decoded_image_batch.permute(0, 2, 3, 1).contiguous()
                decoded_image_batch = decoded_image_batch.to(torch.float32).clamp(0, 1)
            else:
                _log(f"Segment {i+1}: DRY RUN - Skipping VAE Decode.")
                num_frames_to_gen = window_frame_count if i == 0 else window_frame_count - merge_frame_count
                decoded_image_batch = torch.zeros((num_frames_to_gen, render_height, render_width, 3), dtype=torch.float32)

            # --- 3f. Internal Color Matching (Optional) ---
            if decoded_image_batch is not None and decoded_image_batch.shape[0] > 0:
                next_reference_frame = decoded_image_batch[-1:]
                reference_for_this_loop = None
                if static_color_reference is not None:
                    reference_for_this_loop = static_color_reference
                elif i > 0 and previous_loop_last_frame is not None:
                    reference_for_this_loop = previous_loop_last_frame
                
                # Only run segment matching if color_match is on AND color_match_lastframe is OFF
                if color_match_method != "Disabled" and color_match_reference_frame == "Sequential" and reference_for_this_loop is not None:
                    _log(f"Segment {i+1}: Applying segment-to-segment color match to the first frame.")
                    first_frame = decoded_image_batch[0:1]
                    rest_of_frames = decoded_image_batch[1:]
                    
                    matched_first_frame = _apply_color_match(
                        target_batch_tensor=first_frame,
                        reference_tensor=reference_for_this_loop,
                        strength=color_match_strength,
                        method=color_match_method
                    )
                    
                    decoded_image_batch = torch.cat([matched_first_frame, rest_of_frames], dim=0)

                previous_loop_last_frame = next_reference_frame
            else:
                 _log("Warning: Skipping frame storage due to missing decoded batch.")
                 current_seed_image = None

            # --- 3g. Store Frames & Update Seed ---
            if decoded_image_batch is not None and decoded_image_batch.shape[0] > 0:
                if i == 0 and color_match_reference_frame == "First Frame":
                    first_frame_for_color_match = decoded_image_batch[0:1]

                if decoded_image_batch.shape[0] >= merge_frame_count and merge_frame_count > 0:
                    current_seed_image = decoded_image_batch[-merge_frame_count:]
                elif decoded_image_batch.shape[0] > 0: current_seed_image = decoded_image_batch[-1:]
                else: _log("Warning: No frames decoded/generated..."); current_seed_image = None
                frames_to_save = decoded_image_batch if i == 0 else decoded_image_batch[merge_frame_count:]
                all_frames_collected.append(frames_to_save.cpu())
            else:
                 _log("Warning: Skipping frame storage due to missing decoded batch.")
                 current_seed_image = None

            _log(f"Segment {i+1}: Cleaning up loop resources.")

            _log(f"--- Finished Loop {i+1} ---")

        # --- 4. Finalize & Return ---
        if not all_frames_collected:
            _log("Error: No frames were collected."); return (torch.zeros((1, 64, 64, 3)),)*2

        _log("Concatenating final video batch...")
        try:
            final_batch = torch.cat(all_frames_collected, dim=0)
            
        except RuntimeError as e:
            _log(f"CRITICAL ERROR: Failed to concatenate final batch. This may be a channel/shape mismatch between segments. {e}")
            _log("This should not happen if the VAE is consistent. Please report this error.")
            _log("Returning a dummy tensor to prevent a crash.")
            return (torch.zeros((1, 64, 64, 3)),)*2

        # --- 4b. Final Frame Color Match (New Logic) ---
        if color_match_method != "Disabled" and final_batch.shape[0] > 1 and color_match_reference_frame in ["Last Frame", "First Frame", "Reference"]:
            reference_frame_for_final_match = None
            if color_match_reference_frame == "Last Frame":
                _log(f"Applying 'Last Frame' color match using frame {final_batch.shape[0]-1} as reference.")
                reference_frame_for_final_match = final_batch[-1:]
            elif color_match_reference_frame == "First Frame":
                _log(f"Applying 'First Frame' color match using frame 0 as reference.")
                reference_frame_for_final_match = first_frame_for_color_match
            elif color_match_reference_frame == "Reference":
                if static_color_reference is not None:
                    _log(f"Applying 'Reference' color match using provided reference image.")
                    reference_frame_for_final_match = static_color_reference
                else:
                    _log(f"Warning: 'Reference' color match selected, but no reference image provided. Skipping final color match.")

            if reference_frame_for_final_match is not None:
                frames_to_match = final_batch[:-1]

                matched_frames = _apply_color_match(
                    target_batch_tensor=frames_to_match,
                    reference_tensor=reference_frame_for_final_match,
                    strength=color_match_strength,
                    method=color_match_method
                )

                final_batch = torch.cat([matched_frames, reference_frame_for_final_match], dim=0)
                _log(f"'{color_match_reference_frame}' color match complete.")
        elif color_match_method != "Disabled" and final_batch.shape[0] <= 1 and color_match_reference_frame in ["Last Frame", "First Frame", "Reference"]:
             _log(f"Warning: '{color_match_reference_frame}' color match enabled, but not enough frames to perform match (<= 1).")


        last_frame = final_batch[-1:] if final_batch.shape[0] > 0 else torch.zeros((1, render_height, render_width, 3))
        log_message = "DRY RUN Finished" if enable_dry_run else "WanVideoLooper finished"
        _log(f"{log_message}. Total frames: {final_batch.shape[0]}")
        return (final_batch, last_frame)