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
    try:
        print(f"[WanVideoLooper] {msg}", flush=True)
    except Exception:
        pass

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
        model_device = next(model.model.parameters()).device; model_dtype = next(model.model.parameters()).dtype
    except Exception:
        model_device = comfy.model_management.intermediate_device(); model_dtype = torch.float32
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
    pos_tokens = clip.tokenize(positive_prompt or ""); pos_data = clip.encode_from_tokens(pos_tokens, return_pooled=True, return_dict=True)
    positive_cond = [(pos_data["cond"], {"pooled_output": pos_data.get("pooled_output")})]
    neg_tokens = clip.tokenize(negative_prompt or ""); neg_data = clip.encode_from_tokens(neg_tokens, return_pooled=True, return_dict=True)
    negative_cond = [(neg_data["cond"], {"pooled_output": neg_data.get("pooled_output")})]; return positive_cond, negative_cond
def _prepare_latent_window(positive_cond, negative_cond, vae, width, height, frame_count, batch_size, seed_image=None, vision_output=None):
    device = comfy.model_management.intermediate_device()
    latent_height = height // 8; latent_width = width // 8; time_dim = ((frame_count - 1) // 4) + 1
    empty_latent = torch.zeros([batch_size, 16, time_dim, latent_height, latent_width], device=device)
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
# NEW Color Match Helper
# ====================================================================================================
def _apply_color_match(target_batch_tensor, reference_tensor, strength=1.0):
    """Applies MKL color matching from reference to target batch."""
    if not COLOR_MATCHER_AVAILABLE:
        _log("Skipping color match: color-matcher library not available.")
        return target_batch_tensor

    if reference_tensor is None or target_batch_tensor is None:
        return target_batch_tensor

    ref_np = reference_tensor.squeeze(0).cpu().numpy().astype(np.float64)
    target_np = target_batch_tensor.cpu().numpy().astype(np.float64)
    batch_size = target_np.shape[0]
    matched_results = []

    _log(f"Applying internal color match (strength: {strength:.2f}) using MKL method.")
    cm = ColorMatcher()

    for i in range(batch_size):
        try:
            target_img = target_np[i]
            transfer_result = cm.transfer(src=target_img, ref=ref_np, method='mkl')
            final_result = target_img + strength * (transfer_result - target_img)
            final_result = np.clip(final_result, 0.0, 1.0)
            matched_results.append(torch.from_numpy(final_result))
        except Exception as e:
            _log(f"Error during color matching frame {i}: {e}. Using original frame.")
            matched_results.append(torch.from_numpy(target_np[i]))

    matched_batch = torch.stack(matched_results).to(device=target_batch_tensor.device, dtype=torch.float32)
    return matched_batch


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
                "enable_motion_cfg": ("BOOLEAN", {"default": False, "tooltip": "Enable a separate CFG for the very first step to control motion."}),
                "cfg_motion_step": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "CFG for the first step ONLY. A value > 1.0 will engage the negative prompt for initial motion guidance."}),
                "cfg_high_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "CFG for the high-noise model (structure/motion). 1.0 disables guidance."}),
                "cfg_low_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "CFG for the low-noise model (details). 1.0 disables guidance."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampling algorithm (e.g., dpmpp_2m_sde_gpu)."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The noise scheduler (e.g., karras)."}),
                "model_switch_point": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001, "tooltip": "Timestep to switch from high-noise to low-noise model. 0.9 (I2V) or 0.875 (T2V) recommended."}),
                "sigma_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.01, "tooltip": "Applies sigma shift to both models. 8.0 is a good default."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise amount. 1.0 means generate a full new image."}),

                "width": ("INT", {"default": 832, "min": 64, "max": 8192, "step": 16, "tooltip": "The width of the output video."}),
                "height": ("INT", {"default": 480, "min": 64, "max": 8192, "step": 16, "tooltip": "The height of the output video."}),
                "frame_merge": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1, "tooltip": "Number of frames to overlap/merge between loops for smooth transitions."}),
                "duration_sec": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1, "tooltip": "Duration of each segment in seconds. 5s is recommended, max 10."}),
                "color_match": ("BOOLEAN", {"default": False, "tooltip": "Apply color matching between segments internally using MKL method."}),
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
        seed, steps, enable_motion_cfg, cfg_motion_step, cfg_high_noise, cfg_low_noise,
        sampler_name, scheduler, model_switch_point, sigma_shift, denoise,
        width, height, frame_merge, duration_sec, enable_dry_run,
        color_match, color_match_strength,
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

        _log("Cloning base models and clip for looping.")
        base_model_high = model_high.clone()
        base_model_low = model_low.clone()
        base_clip = clip.clone()

        _log(f"Starting loop with {len(positive_prompts)} prompts...")
        if enable_dry_run: _log("!!! DRY RUN MODE ENABLED - Sampling and decoding will be skipped !!!")
        if color_match and not COLOR_MATCHER_AVAILABLE:
             _log("Warning: Internal color match enabled, but 'color-matcher' library is missing. Feature disabled.")
             color_match = False

        model_device, model_dtype = _get_model_device_info(base_model_high)
        cpu_device = torch.device("cpu")
        render_width = _align_to_multiple(width, 16)
        render_height = _align_to_multiple(height, 16)
        current_seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        total_steps = int(steps)

        # --- Color Match Setup ---
        static_color_reference = color_match_ref[0:1] if color_match_ref is not None else None
        previous_loop_last_frame = None

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
            _log(f"Segment {i+1}: Cloning fresh base models and clip.")
            loop_model_high = base_model_high.clone()
            loop_model_low = base_model_low.clone()
            loop_clip = base_clip.clone()
            segment_models_provided = False
            using_provided_high, using_provided_low, using_provided_clip = False, False, False
            if valid_sequence and i < len(model_clip_sequence):
                segment_data = model_clip_sequence[i]
                if segment_data and isinstance(segment_data, tuple) and len(segment_data) == 3:
                    mh, ml, c = segment_data
                    if mh is not None: loop_model_high = mh; segment_models_provided = True; using_provided_high = True
                    if ml is not None: loop_model_low = ml; segment_models_provided = True; using_provided_low = True
                    if c is not None: loop_clip = c; segment_models_provided = True; using_provided_clip = True

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
                current_seed_image, vision_output
            )
            latent_on_gpu = {"samples": _move_tensor(latent_on_cpu["samples"], model_device, model_dtype)}
            final_latent = None

            # --- 3c/3d. Run Samplers OR Skip for Dry Run ---
            if not enable_dry_run:
                _log(f"Segment {i+1}: Running HIGH noise sampler (Steps 0 to {switch_step_index}).")
                if enable_motion_cfg and switch_step_index > 1:
                    _log(f"Segment {i+1}: Using special Motion CFG ({cfg_motion_step}) for step 0.")
                    initial_latent = _run_sampler_step(loop_model_high, current_seed, total_steps, cfg_motion_step, sampler_name, scheduler, loop_positive_cond, loop_negative_cond, latent_on_gpu, denoise, disable_noise=False, start_step=0, last_step=1, force_full_denoise=False)
                    high_noise_latent = _run_sampler_step(loop_model_high, current_seed, total_steps, cfg_high_noise, sampler_name, scheduler, loop_positive_cond, loop_negative_cond, initial_latent, denoise, disable_noise=True, start_step=1, last_step=switch_step_index, force_full_denoise=False)
                else:
                    high_noise_latent = _run_sampler_step(loop_model_high, current_seed, total_steps, cfg_high_noise, sampler_name, scheduler, loop_positive_cond, loop_negative_cond, latent_on_gpu, denoise, disable_noise=False, start_step=0, last_step=switch_step_index, force_full_denoise=False)

                _log(f"Segment {i+1}: Running LOW noise sampler (Steps {switch_step_index} to {total_steps}).")
                final_latent = _run_sampler_step(loop_model_low, current_seed, total_steps, cfg_low_noise, sampler_name, scheduler, loop_positive_cond, loop_negative_cond, high_noise_latent, denoise, disable_noise=True, start_step=switch_step_index, last_step=total_steps, force_full_denoise=True)
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
                if color_match and reference_for_this_loop is not None:
                    decoded_image_batch = _apply_color_match(
                        target_batch_tensor=decoded_image_batch,
                        reference_tensor=reference_for_this_loop,
                        strength=color_match_strength
                    )

                previous_loop_last_frame = next_reference_frame
            else:
                 previous_loop_last_frame = None

            # --- 3g. Store Frames & Update Seed ---
            if decoded_image_batch is not None and decoded_image_batch.shape[0] > 0:
                if decoded_image_batch.shape[0] >= merge_frame_count and merge_frame_count > 0:
                    current_seed_image = decoded_image_batch[-merge_frame_count:-merge_frame_count+1 if merge_frame_count > 1 else None]
                elif decoded_image_batch.shape[0] > 0: current_seed_image = decoded_image_batch[-1:]
                else: _log("Warning: No frames decoded/generated..."); current_seed_image = None
                frames_to_save = decoded_image_batch if i == 0 else decoded_image_batch[merge_frame_count:]
                all_frames_collected.append(frames_to_save.cpu())
            else:
                 _log("Warning: Skipping frame storage due to missing decoded batch.")
                 current_seed_image = None

            # --- 3h. Per-Iteration Cleanup ---
            _log(f"Segment {i+1}: Cleaning up loop resources.")
            del loop_model_high, loop_model_low, loop_clip
            _log(f"Segment {i+1}: Deleted loop model/clip variables.")
            if 'high_noise_latent' in locals(): del high_noise_latent
            if final_latent is not None: del final_latent
            if 'latent_on_gpu' in locals(): del latent_on_gpu
            if 'final_latent_on_cpu' in locals(): del final_latent_on_cpu
            if 'decoded_data' in locals(): del decoded_data
            if decoded_image_batch is not None: del decoded_image_batch
            if 'frames_to_save' in locals(): del frames_to_save
            _log(f"Segment {i+1}: Forcing garbage collection.")
            gc.collect()
            if torch.cuda.is_available():
                _log(f"Segment {i+1}: Emptying CUDA cache.")
                torch.cuda.memory.empty_cache()
            _log(f"--- Finished Loop {i+1} ---")

        # --- 4. Finalize & Return ---
        if not all_frames_collected:
            _log("Error: No frames were collected."); return (torch.zeros((1, 64, 64, 3)),)*2
        _log("Concatenating final video batch.")
        num_channels = all_frames_collected[0].shape[-1]
        consistent_batches = []
        for batch in all_frames_collected:
             if batch.shape[-1] == num_channels: consistent_batches.append(batch)
             else:
                 _log(f"Warning: Channel mismatch found (Expected {num_channels}, Got {batch.shape[-1]}). Adjusting batch.")
                 if batch.shape[-1] == 1 and num_channels == 3: batch = batch.repeat(1, 1, 1, 3)
                 elif batch.shape[-1] == 4 and num_channels == 3: batch = batch[:, :, :, :3]
                 elif batch.shape[-1] == 3 and num_channels == 4: alpha = torch.ones_like(batch[:, :, :, :1]); batch = torch.cat([batch, alpha], dim=-1)
                 else: batch = torch.zeros((batch.shape[0], batch.shape[1], batch.shape[2], num_channels), dtype=batch.dtype)
                 consistent_batches.append(batch)

        final_batch = torch.cat(consistent_batches, dim=0)
        last_frame = final_batch[-1:] if final_batch.shape[0] > 0 else torch.zeros((1, render_height, render_width, 3))
        log_message = "DRY RUN Finished" if enable_dry_run else "WanVideoLooper finished"
        _log(f"{log_message}. Total frames: {final_batch.shape[0]}")
        return (final_batch, last_frame)