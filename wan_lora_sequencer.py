import folder_paths

# ====================================================================================================
# Logging Utility
# ====================================================================================================
def _log(msg: str):
    """Simple print logger for the console."""
    try:
        print(f"[WanVideoLooper:Sequencer] {msg}", flush=True)
    except Exception:
        pass

# ====================================================================================================
# WanVideo Lora Sequencer Node
# ====================================================================================================
class WanVideoLoraSequencer:

    @classmethod
    def INPUT_TYPES(s):
        inputs = {"optional": {}}
        for i in range(1, 11): 
            inputs["optional"][f"model_high_{i}"] = ("MODEL", {"tooltip": f"Patched HIGH model for segment {i}."})
        for i in range(1, 11):
            inputs["optional"][f"model_low_{i}"] = ("MODEL", {"tooltip": f"Patched LOW model for segment {i}."})
        for i in range(1, 11):
            inputs["optional"][f"clip_{i}"] = ("CLIP", {"tooltip": f"Patched CLIP model for segment {i}."})
        return inputs

    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("model_clip_sequence",)
    FUNCTION = "sequence_models_clips"
    CATEGORY = "WanVideoLooper"

    def sequence_models_clips(self, **kwargs):
        """Creates a list of tuples, each containing models/clip for a segment."""
        model_clip_sequence = []
        for i in range(1, 11):
            model_high = kwargs.get(f"model_high_{i}", None)
            model_low = kwargs.get(f"model_low_{i}", None)
            clip = kwargs.get(f"clip_{i}", None)

            segment_data = (model_high, model_low, clip)
            model_clip_sequence.append(segment_data)

            if any(segment_data):
                 _log(f"Found patched models/clip for segment {i}")

        return (model_clip_sequence, )