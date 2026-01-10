# ComfyUI-QwenVL-MultiImage AI Coding Guide

## Project Overview

This is a **ComfyUI custom node** that integrates Qwen2.5-VL and Qwen3-VL vision-language models with multi-image support. It's a plugin for ComfyUI's node-based image generation workflow system.

**Core Architecture:**
- `nodes.py`: Two node classes (`QwenVL_MultiImage` and `QwenVL_MultiImage_Advanced`) that process images using HuggingFace transformers
- `__init__.py`: Node registration with ComfyUI (exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`)
- Global model cache (`_MODEL_CACHE`) for keeping models loaded in VRAM between inferences

## ComfyUI Node Development Patterns

### Node Class Structure
```python
class YourNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}, "optional": {...}}
    
    RETURN_TYPES = ("STRING",)  # Tuple of output types
    RETURN_NAMES = ("text",)    # Tuple of output names
    FUNCTION = "method_name"     # Method to call on execution
    CATEGORY = "🧪AILab/QwenVL"  # Node menu location
    
    def method_name(self, ...):  # Must match FUNCTION
        return (result,)          # Must be tuple matching RETURN_TYPES
```

### Image Tensor Handling
ComfyUI images are `torch.Tensor` in format `[H, W, C]` or `[B, H, W, C]` with values in `[0, 1]`:
- Always check `tensor.ndim` to distinguish single (3D) vs batch (4D)
- Convert to PIL: multiply by 255, convert to uint8, use `Image.fromarray()`
- Function `tensor_to_pil()` in nodes.py handles this pattern

### Multi-Image Input Pattern
This project uses a specific pattern for accepting multiple image batches:
- `images`: Required main input (can be batch)
- `images_batch_2`, `images_batch_3`: Optional additional batches
- Loop through all batches, extract individual images, append to `all_images` list
- See `generate()` methods for the complete pattern

## Critical Integration Points

### HuggingFace Transformers Integration
- Use `AutoProcessor` and `AutoModelForVision2Seq` (not `AutoModel`) for Qwen-VL
- Message format: `[{"role": "system/user", "content": [{"type": "image/text", ...}]}]`
- Must call `process_vision_info()` from `qwen_vl_utils` to extract image/video inputs
- Apply chat template before processing: `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`

### Quantization System
Three modes implemented via BitsAndBytesConfig:
- `None (FP16)`: `torch_dtype=torch.float16, device_map="auto"`
- `8-bit (Balanced)`: `BitsAndBytesConfig(load_in_8bit=True)`
- `4-bit (VRAM-friendly)`: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`
- FP8 models (pre-quantized): Detect by "FP8" in model name, use `torch.float8_e4m3fn` if available

### Model Caching Strategy
Global cache key: `f"{model_name}_{quantization}_{device}"`
- Check cache before loading: `if cache_key in _MODEL_CACHE: return _MODEL_CACHE[cache_key]`
- Cache stores tuple: `(model, processor)`
- Clear with `clear_model_cache()` which also calls `torch.cuda.empty_cache()`
- User controls via `keep_model_loaded` parameter

## Development Workflows

### Adding New Node Parameters
1. Add to `INPUT_TYPES()` in both `required` or `optional` dicts
2. Add to method signature (order matters!)
3. Use parameter in generation kwargs or model loading
4. Update README.md parameter table

### Adding New Models
1. Add model identifier to `QWEN_MODELS` list in nodes.py
2. No code changes needed - autodetects Qwen2.5-VL vs Qwen3-VL
3. FP8 models detected by "FP8" substring in name

### Debugging VRAM Issues
- Check `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`
- Verify `clear_model_cache()` is called when `keep_model_loaded=False`
- Test with smaller model (2B or 3B) and 4-bit quantization first

## Project-Specific Conventions

### Error Handling Philosophy
- Let transformers library handle model loading errors with full stack traces
- Print informative messages before slow operations: `print(f"Loading model: {model_name} with {quantization}")`
- No custom exception catching in node code - ComfyUI shows errors in UI

### Dependency Management
- Core deps: `transformers>=4.45.0`, `torch>=2.0.0`, `qwen-vl-utils>=0.0.8`
- Optional: `flash-attn` (checked at runtime, gracefully falls back)
- `bitsandbytes>=0.41.0` required for quantization

### File Organization
- All node logic in `nodes.py` (no separate utils or helpers)
- `__init__.py` only for registration (keep minimal)
- Example workflows in `example_workflows/` as `.json` files
- Documentation split: README.md (users), PROJECT_SUMMARY.md (developers), QUICKSTART.md (new users)

## Common Patterns to Follow

### Generation Parameter Handling
```python
gen_kwargs = {"max_new_tokens": max_tokens, "repetition_penalty": repetition_penalty}
if num_beams == 1:
    gen_kwargs["do_sample"] = True  # Only enable sampling without beam search
    gen_kwargs.update({"temperature": temperature, "top_p": top_p})
```

### Trimming Output Tokens
Always remove input tokens from generated output:
```python
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
```

### Flash Attention 2 Detection
```python
try:
    import flash_attn
    load_kwargs["attn_implementation"] = "flash_attention_2"
except ImportError:
    pass  # Use default attention
```

## Testing Checklist
Before committing changes:
- [ ] Test with single image (3D tensor)
- [ ] Test with batch of images (4D tensor)
- [ ] Test with multiple optional batches
- [ ] Verify both Standard and Advanced nodes
- [ ] Test different quantization modes
- [ ] Check model caching works (second run faster)
- [ ] Ensure `keep_model_loaded=False` clears VRAM

## Model Support Matrix
- **Qwen3-VL**: 4B, 8B, 32B (Instruct + Thinking variants, FP8 versions)
- **Qwen2.5-VL**: 2B, 3B, 7B, 72B (Instruct only)
- All models auto-download from HuggingFace on first use
- Use `AutoModelForVision2Seq` for compatibility with both series
