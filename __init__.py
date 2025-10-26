from .wan_looper_node import WanVideoLooper
from .wan_lora_sequencer import WanVideoLoraSequencer
from .wan_prompts_node import WanVideoLooperPrompts

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "WanVideoLooper": WanVideoLooper,
    "WanVideoLoraSequencer": WanVideoLoraSequencer,
    "WanVideoLooperPrompts": WanVideoLooperPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoLooper": "WanVideo Looper",
    "WanVideoLoraSequencer": "WanVideo Looper Lora Sequencer",
    "WanVideoLooperPrompts": "WanVideo Looper Prompts",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("Loaded 'WanVideoLooper' custom nodes.")