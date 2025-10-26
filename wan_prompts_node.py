import json

# ====================================================================================================
# Logging Utility
# ====================================================================================================
def _log(msg: str):
    """Simple print logger for the console."""
    try:
        print(f"[WanVideoLooper:Prompts] {msg}", flush=True)
    except Exception:
        pass

# ====================================================================================================
# WanVideo Looper Prompts Node (Simple Multiline)
# ====================================================================================================
class WanVideoLooperPrompts:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True, "default": "", "tooltip": "Enter one positive prompt per line. Empty lines are ignored."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Enter the global negative prompt."}),
                "enable_prefix_suffix": (["true", "false"], {"default": "true", "tooltip": "Enable the prefix and suffix string."}),
                "prefix": ("STRING", {"default": "", "tooltip": "Text to add BEFORE each prompt line (positive and negative)."}),
                "suffix": ("STRING", {"default": "", "tooltip": "Text to add AFTER each prompt line (positive and negative)."}),
            }
        }

    RETURN_TYPES = ("*", "STRING",)
    RETURN_NAMES = ("prompt_list", "negative_prompt",)
    OUTPUT_IS_LIST = (False, False,)
    FUNCTION = "process_multiline_prompts"
    CATEGORY = "WanVideoLooper"

    def _process_single_prompt(self, prompt_text, prefix, suffix, enable_prefix_suffix):
        """Applies prefix/suffix to a single stripped prompt line."""
        prompt_text = prompt_text.strip()
        
        if not prompt_text:
            return ""

        is_prefix_suffix_enabled = (enable_prefix_suffix == "true")

        prefix_text = prefix.strip() if is_prefix_suffix_enabled else ""
        suffix_text = suffix.strip() if is_prefix_suffix_enabled else ""

        parts = []
        if prefix_text: parts.append(prefix_text)
        parts.append(prompt_text)
        if suffix_text: parts.append(suffix_text)

        return ",".join(parts)

    def process_multiline_prompts(self, prompts, negative_prompt, enable_prefix_suffix, prefix, suffix):
        """Processes multiline prompts, applies prefix/suffix, and filters empty lines."""
        
        positive_prompt_list = []
        lines = prompts.replace('\r\n', '\n').split('\n')

        for line in lines:
            stripped_line = line.strip()
            
            if stripped_line:
                processed_line = self._process_single_prompt(
                    stripped_line, prefix, suffix, enable_prefix_suffix
                )
                positive_prompt_list.append(processed_line)

        processed_negative = self._process_single_prompt(
            negative_prompt, prefix, suffix, enable_prefix_suffix
        )

        _log(f"Processed {len(positive_prompt_list)} positive prompts.")
        
        if not positive_prompt_list:
             _log("Warning: No valid positive prompts found. Using one empty prompt.")
             positive_prompt_list.append("")


        return (positive_prompt_list, processed_negative,)

NODE_CLASS_MAPPINGS = {
    "WanVideoLooperPrompts": WanVideoLooperPrompts
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoLooperPrompts": "WanVideo Looper Prompts"
}