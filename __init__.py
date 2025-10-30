"""
ComfyUI QwenVL Multi-Image Node
Supports Qwen2.5-VL and Qwen3-VL models with multiple image inputs
"""

from .nodes import QwenVL_MultiImage, QwenVL_MultiImage_Advanced

NODE_CLASS_MAPPINGS = {
    "QwenVL_MultiImage": QwenVL_MultiImage,
    "QwenVL_MultiImage_Advanced": QwenVL_MultiImage_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL_MultiImage": "🧪 QwenVL Multi-Image",
    "QwenVL_MultiImage_Advanced": "🧪 QwenVL Multi-Image (Advanced)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

