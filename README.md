# WanVideoLooper Nodes for ComfyUI

This is a set of custom nodes for ComfyUI designed for generating sequential long video clips based on the Wan 2.2 model architecture, handling continuity between segments and offering advanced control over the sampling process, including per-segment LoRA/model application.

This pack contains two nodes:

---

1.  **`WanVideo Looper`**: The main node that takes base models, a list of prompts, and various settings to generate a sequence of video clips, merging them seamlessly.
    

---

2.  **`WanVideo Looper Lora Sequencer`**: A helper node designed to feed pre-patched Model and CLIP objects into the `WanVideo Looper` for specific segments, enabling per-segment LoRA application.
    

---

## How to Install

1.  Navigate to your ComfyUI installation directory.
2.  Go into the `ComfyUI/custom_nodes/` folder.
3.  Create a new folder inside `custom_nodes` named **`WanVideoLooper`**.
4.  Place all three files (`wan_looper_node.py`, `wan_lora_sequencer.py`, and `__init__.py`) inside the new `WanVideoLooper` folder.
5.  **Restart your ComfyUI server.**

The nodes will be available by right-clicking the canvas and selecting **Add Node > WanVideoLooper**.

---

## 1. WanVideo Looper

This is the core node for generating video sequences segment by segment. It intelligently handles continuity and uses an advanced MoE-style sampling approach suitable for Wan 2.2 dual-model setups.

### Features

* **In-Memory Looping:** Processes segments entirely in memory for efficiency.
* **Prompt List Input:** Takes a list of positive prompts (e.g., from the `MultiString Prompt` node), generating one segment per prompt.
* **Seamless Continuity:** Uses frame merging (`frame_merge`) to ensure smooth transitions between segments based on the last frame(s) of the previous segment.
* **MoE-Style Sampling:** Integrates Mixture of Experts logic with a single `steps` input and automatic calculation of the split point (`model_switch_point`) between high-noise and low-noise models.
* **Advanced Controls:** Includes `sigma_shift`, separate `cfg_high_noise` / `cfg_low_noise`, and an optional `cfg_motion_step` for fine-tuning the first step's guidance.
* **Variable Duration:** Allows setting segment duration (`duration_sec`), although 5 seconds is recommended for model consistency.
* **Per-Segment Models/LoRAs:** Supports an optional input from the `WanVideo Lora Sequencer` to use completely different, pre-patched models (with LoRAs applied) for specific segments.
* **Dry Run Mode:** Includes an `enable_dry_run` toggle to quickly test setup and logs without performing sampling/decoding.

### Inputs & Outputs

* **Required Inputs**
    * `model_high`: The BASE high-noise model (can be pre-patched with global LoRAs).
    * `model_low`: The BASE low-noise model (can be pre-patched with global LoRAs).
    * `clip`: The BASE CLIP model (can be pre-patched with global LoRAs).
    * `vae`: The VAE model used for decoding the final image.
    * `start_image`: The seed image that the first loop will start from.
    * `positive_prompts`: Connect the 'prompt_list' output from the MultiString Prompt node.
    * `negative_prompt`: A single, global negative prompt applied to all loops.
    * `seed`: The random seed. This will be used for all loops unless overridden by future features.
    * `steps`: The total number of sampling steps for each loop.
    * `enable_motion_cfg`: Enable a separate CFG for the very first step to control motion.
    * `cfg_motion_step`: CFG for the first step ONLY. A value > 1.0 will engage the negative prompt for initial motion guidance.
    * `cfg_high_noise`: CFG for the high-noise model (structure/motion). 1.0 disables guidance.
    * `cfg_low_noise`: CFG for the low-noise model (details). 1.0 disables guidance.
    * `sampler_name`: The sampling algorithm (e.g., dpmpp_2m_sde_gpu).
    * `scheduler`: The noise scheduler (e.g., karras).
    * `model_switch_point`: Timestep to switch from high-noise to low-noise model. 0.9 (I2V) or 0.875 (T2V) recommended.
    * `sigma_shift`: Applies sigma shift to both models. 8.0 is a good default.
    * `denoise`: Denoise amount. 1.0 means generate a full new image.
    * `width`: The width of the output video.
    * `height`: The height of the output video.
    * `frame_merge`: Number of frames to overlap/merge between loops for smooth transitions.
    * `duration_sec`: Duration of each segment in seconds. 5s is recommended, max 10.
    * `enable_dry_run`: If enabled, skips sampling and decoding to quickly check setup and logs.
* **Optional Inputs**
    * `clip_vision`: (Optional) A CLIP Vision model for guiding the start image.
    * `model_clip_sequence`: (Optional) Connect a WanVideo Lora Sequencer to use different pre-patched models/clips per segment.
* **Outputs**
    * `images`: The final, concatenated batch of all generated video frames.
    * `last_frame`: The single, very last frame of the entire sequence. Useful for reference in subsequent nodes (e.g., Color Match).

---

## 2. WanVideo Looper Lora Sequencer

This helper node allows you to specify different, pre-patched Model and CLIP objects for each segment of your video sequence. It's the key to applying different LoRAs to different parts of your video when using the `WanVideo Looper`.

### What It Does

* **Segment Inputs:** Provides 5 sets of optional inputs (`model_high_X`, `model_low_X`, `clip_X`) corresponding to the 5 potential prompt segments.
* **Uses Standard Nodes:** You patch your models using standard ComfyUI nodes (`Load LoRA`, `Lora Stacker`, `Power Lora`, etc.) outside the sequencer.
* **Clean Looper:** You connect the *output* of your patching nodes (the patched MODEL and CLIP) into the corresponding inputs on this sequencer. This keeps the main `WanVideo Looper` node uncluttered.
* **Bundles Inputs:** The sequencer simply bundles these potentially connected models/clips into a list.
* **Fallback:** If you leave the inputs for a specific segment unconnected on the sequencer, the `WanVideo Looper` will automatically fall back to using the clean base models for that segment.

### Inputs & Outputs

* **Optional Inputs (15 total)**
    * `model_high_1` to `model_high_5`: The pre-patched HIGH model for each segment.
    * `model_low_1` to `model_low_5`: The pre-patched LOW model for each segment.
    * `clip_1` to `clip_5`: The pre-patched CLIP model for each segment.
* **Outputs**
    * `model_clip_sequence`: A list containing 5 tuples. Each tuple holds `(model_high, model_low, clip)` for the corresponding segment, or `None` if an input was not connected. This output connects directly to the `model_clip_sequence` input on the `WanVideo Looper`.

---