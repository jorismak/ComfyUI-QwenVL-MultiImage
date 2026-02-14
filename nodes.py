"""
ComfyUI QwenVL Multi-Image Nodes
Main implementation of QwenVL nodes with multi-image support
"""

import torch
import gc
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
import folder_paths

# Model definitions
QWEN_MODELS = [
    # Qwen3-VL models
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct-FP8",
    "Qwen/Qwen3-VL-32B-Instruct-FP8",
    "Qwen/Qwen3-VL-8B-Thinking-FP8",
    "Qwen/Qwen3-VL-32B-Thinking-FP8",
    # Qwen2.5-VL models
    "Qwen/Qwen2.5-VL-2B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
]

QUANTIZATION_OPTIONS = [
    "None (As-is)",
    "None (FP16)",
    "None (BF16)",
    "8-bit (Balanced)",
    "4-bit (VRAM-friendly)",
]

# Global model cache
_MODEL_CACHE = {}


def clear_model_cache():
    """Clear all cached models from memory"""
    global _MODEL_CACHE
    for key in list(_MODEL_CACHE.keys()):
        del _MODEL_CACHE[key]
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI image tensor to PIL Image
    ComfyUI format: [H, W, C] with values in [0, 1]
    """
    # Ensure tensor is on CPU and convert to numpy
    if tensor.ndim == 4:
        # Batch of images, take first one
        tensor = tensor[0]

    numpy_image = tensor.cpu().numpy()

    # Convert from [0, 1] to [0, 255]
    numpy_image = (numpy_image * 255).astype(np.uint8)

    # Create PIL image
    return Image.fromarray(numpy_image)


def load_model(model_name: str, quantization: str, device: str = "auto"):
    """
    Load Qwen VL model with specified quantization
    Returns (model, processor)
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    cache_key = f"{model_name}_{quantization}_{device}"

    # Return cached model if available
    if cache_key in _MODEL_CACHE:
        print(f"Using cached model: {model_name}")
        return _MODEL_CACHE[cache_key]

    print(f"Loading model: {model_name} with {quantization}")

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    if device != "cuda" and quantization in {
        "4-bit (VRAM-friendly)",
        "8-bit (Balanced)",
    }:
        print(
            "8-bit/4-bit quantization requires CUDA (bitsandbytes). "
            "Falling back to FP16."
        )
        quantization = "None (FP16)"

    # Check if model is pre-quantized (FP8)
    is_fp8_model = "FP8" in model_name

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Configure quantization
    load_kwargs = {}

    if is_fp8_model:
        # FP8 models are pre-quantized and should be loaded as-is
        print(f"Loading pre-quantized FP8 model (checkpoint-native dtype)")
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
    elif quantization == "None (As-is)":
        print("Loading model with checkpoint-native dtype")
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
    elif quantization == "4-bit (VRAM-friendly)":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["device_map"] = "auto"
    elif quantization == "8-bit (Balanced)":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["device_map"] = "auto"
    elif quantization == "None (BF16)":
        load_kwargs["torch_dtype"] = torch.bfloat16
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
    else:  # None (FP16)
        load_kwargs["torch_dtype"] = torch.float16
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

    # Enable Flash Attention 2 if available (must be set before loading)
    try:
        import flash_attn

        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("Flash Attention 2 enabled")
    except ImportError:
        print("Flash Attention 2 not available, using default attention")

    # Load model using AutoModelForVision2Seq to support both Qwen2-VL and Qwen3-VL
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    if device != "cuda":
        model = model.to(device)

    # Cache the model
    _MODEL_CACHE[cache_key] = (model, processor)

    return model, processor


class QwenVL_MultiImage:
    """
    Standard QwenVL node with multi-image support
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": (QWEN_MODELS, {"default": "Qwen/Qwen3-VL-4B-Instruct"}),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": "You are a helpful assistant."},
                ),
                "user_prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe these images in detail."},
                ),
                "quantization": (QUANTIZATION_OPTIONS, {"default": "8-bit (Balanced)"}),
                "max_tokens": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 64},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "images_batch_2": ("IMAGE",),
                "images_batch_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/QwenVL"

    def generate(
        self,
        images,
        model_name,
        system_prompt,
        user_prompt,
        quantization,
        max_tokens,
        keep_model_loaded,
        seed,
        images_batch_2=None,
        images_batch_3=None,
    ):
        """
        Generate text from multiple images and prompts
        """
        from qwen_vl_utils import process_vision_info

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Load model
        model, processor = load_model(model_name, quantization)

        # Collect all images
        all_images = []

        # Process main image batch
        if images.ndim == 4:
            # Multiple images in batch
            for i in range(images.shape[0]):
                pil_image = tensor_to_pil(images[i])
                all_images.append(pil_image)
        else:
            # Single image
            pil_image = tensor_to_pil(images)
            all_images.append(pil_image)

        # Process optional image batches
        for optional_batch in [images_batch_2, images_batch_3]:
            if optional_batch is not None:
                if optional_batch.ndim == 4:
                    for i in range(optional_batch.shape[0]):
                        pil_image = tensor_to_pil(optional_batch[i])
                        all_images.append(pil_image)
                else:
                    pil_image = tensor_to_pil(optional_batch)
                    all_images.append(pil_image)

        print(f"Processing {len(all_images)} image(s)")

        # Build message content with multiple images
        content = []
        for img in all_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_prompt})

        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # Process messages
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if "Qwen3-VL" in model_name:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadatas = list(video_metadatas)
            else:
                video_metadatas = None

            # Prepare inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )

            # Prepare inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        inputs = inputs.to(model.device)

        # Generate
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
            )

        # Trim the input tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode output
        output_text = processor.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Clean up if not keeping model loaded
        if not keep_model_loaded:
            clear_model_cache()

        return (output_text,)


class QwenVL_MultiImage_Advanced:
    """
    Advanced QwenVL node with full parameter control and multi-image support
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": (QWEN_MODELS, {"default": "Qwen/Qwen3-VL-4B-Instruct"}),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": "You are a helpful assistant."},
                ),
                "user_prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe these images in detail."},
                ),
                "quantization": (QUANTIZATION_OPTIONS, {"default": "8-bit (Balanced)"}),
                "max_tokens": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 64},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1},
                ),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "device": (["auto", "cuda", "xpu", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "images_batch_2": ("IMAGE",),
                "images_batch_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/QwenVL"

    def generate(
        self,
        images,
        model_name,
        system_prompt,
        user_prompt,
        quantization,
        max_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        num_beams,
        keep_model_loaded,
        seed,
        device,
        images_batch_2=None,
        images_batch_3=None,
    ):
        """
        Generate text from multiple images and prompts with advanced control
        """
        from qwen_vl_utils import process_vision_info

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Load model
        model, processor = load_model(model_name, quantization, device)

        # Collect all images
        all_images = []

        # Process main image batch
        if images.ndim == 4:
            # Multiple images in batch
            for i in range(images.shape[0]):
                pil_image = tensor_to_pil(images[i])
                all_images.append(pil_image)
        else:
            # Single image
            pil_image = tensor_to_pil(images)
            all_images.append(pil_image)

        # Process optional image batches
        for optional_batch in [images_batch_2, images_batch_3]:
            if optional_batch is not None:
                if optional_batch.ndim == 4:
                    for i in range(optional_batch.shape[0]):
                        pil_image = tensor_to_pil(optional_batch[i])
                        all_images.append(pil_image)
                else:
                    pil_image = tensor_to_pil(optional_batch)
                    all_images.append(pil_image)

        print(f"Processing {len(all_images)} image(s) with advanced settings")

        # Build message content with multiple images
        content = []
        for img in all_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_prompt})

        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # Process messages
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if "Qwen3-VL" in model_name:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadatas = list(video_metadatas)
            else:
                video_metadatas = None

            # Prepare inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )

            # Prepare inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        inputs = inputs.to(model.device)

        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
        }

        # Add sampling parameters only if not using beam search
        if num_beams == 1:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        # Generate
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **gen_kwargs)

        # Trim the input tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode output
        output_text = processor.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Clean up if not keeping model loaded
        if not keep_model_loaded:
            clear_model_cache()

        return (output_text,)
