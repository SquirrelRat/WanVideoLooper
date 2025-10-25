from .wan_looper_node import WanVideoLooper
from .wan_lora_sequencer import WanVideoLoraSequencer

NODE_CLASS_MAPPINGS = {
    "WanVideoLooper": WanVideoLooper,
    "WanVideoLoraSequencer": WanVideoLoraSequencer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoLooper": "WanVideo Looper",
    "WanVideoLoraSequencer": "WanVideo Looper Lora Sequencer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("Loaded 'WanVideoLooper' custom nodes.")