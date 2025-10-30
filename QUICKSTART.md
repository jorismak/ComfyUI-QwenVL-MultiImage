# Quick Start Guide

Get up and running with ComfyUI-QwenVL-MultiImage in 5 minutes!

## 🚀 Installation

### Step 1: Install the Node

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/YOUR_USERNAME/ComfyUI-QwenVL-MultiImage.git
cd ComfyUI-QwenVL-MultiImage
pip install -r requirements.txt
```

### Step 2: Restart ComfyUI

Close and restart ComfyUI to load the new nodes.

## 🎯 First Use

### Single Image Analysis

1. In ComfyUI, add these nodes:
   - `Load Image` → Connect to `🧪 QwenVL Multi-Image` → Connect to `Show Text`

2. Configure the QwenVL node:
   - **Model**: `Qwen/Qwen3-VL-4B-Instruct` (good starting point)
   - **User Prompt**: "Describe this image in detail"
   - **Quantization**: `8-bit (Balanced)`
   - Leave other settings as default

3. Click **Queue Prompt**

4. **First run will take 2-5 minutes** (downloads model from HuggingFace)

5. Subsequent runs will be much faster!

### Multiple Image Comparison

1. Load 2-3 images using `Load Image` nodes

2. Connect them to `🧪 QwenVL Multi-Image`:
   - First image → `images`
   - Second image → `images_batch_2`
   - Third image → `images_batch_3`

3. Set **User Prompt**: "Compare these images and describe their differences"

4. Run!

## 💡 Quick Tips

### Choose the Right Model

| VRAM | Recommended Model | Quantization |
|------|------------------|--------------|
| 6GB  | Qwen2.5-VL-2B-Instruct | 4-bit |
| 8GB  | Qwen3-VL-4B-Instruct | 8-bit |
| 12GB | Qwen3-VL-7B-Instruct | 8-bit |
| 16GB+ | Qwen3-VL-8B-Instruct | FP16 |
| 24GB+ | Qwen3-VL-32B-Instruct | 8-bit or FP16 |

### Performance Settings

**For Speed:**
- Keep `keep_model_loaded` = True
- Use 8-bit or FP16 quantization
- Use FP8 models if you have RTX 40 series GPU

**For Memory:**
- Use 4-bit quantization
- Use smaller models (2B, 3B, 4B)
- Set `keep_model_loaded` = False

### Good Prompts

✅ **Specific prompts work best:**
- "Compare the color schemes and composition of these images"
- "Describe the architectural style shown in these photos"
- "List the differences between image 1 and image 2"

❌ **Avoid vague prompts:**
- "Describe this"
- "What do you see"
- "Tell me about these"

## 🐛 Troubleshooting

### Out of Memory Error
→ Switch to smaller model or 4-bit quantization

### Slow First Run
→ Normal! Model is downloading (2-30GB depending on model)

### Can't Find Node
→ Restart ComfyUI, check `custom_nodes` directory

### Import Errors
→ Run: `pip install -r requirements.txt --upgrade`

## 📖 Need More Help?

- **Full Documentation**: See [README.md](README.md)
- **Example Workflows**: Check `example_workflows/` folder
- **Test Installation**: Run `python test_installation.py`

## 🎓 Example Prompts

### Image Analysis
```
"Provide a detailed description of this image, including objects, colors, composition, and mood."
```

### Comparison
```
"Compare these images focusing on: 1) visual style, 2) subject matter, 3) color palette, 4) overall impression."
```

### Technical Analysis
```
"Analyze the technical aspects of these photos including exposure, composition, depth of field, and lighting."
```

### Creative Writing
```
"Write a short story inspired by these images, connecting the scenes they depict."
```

## 🎨 Advanced Usage

Want more control? Use the **Advanced Node**:

- `🧪 QwenVL Multi-Image (Advanced)`

Additional parameters:
- `temperature`: 0.7 (creativity level)
- `top_p`: 0.9 (diversity)
- `num_beams`: 1 (beam search for better quality)
- `device`: "auto" (or force cuda/cpu)

## ⚡ Pro Tips

1. **Batch Processing**: Connect multiple images to same node input for batch analysis

2. **Model Caching**: First model load is slow, keep `keep_model_loaded=True` for speed

3. **System Prompts**: Set the role for better results:
   ```
   "You are an expert art critic analyzing paintings"
   "You are a technical photographer"
   "You are a medical imaging specialist"
   ```

4. **Chain Workflows**: Use the text output with other nodes for creative workflows

5. **GPU Memory**: Monitor VRAM usage in Task Manager/Activity Monitor

## 🎯 Next Steps

- Try the example workflows in `example_workflows/`
- Experiment with different models
- Read the full [README.md](README.md) for all features
- Join ComfyUI community to share your workflows!

---

Ready to create something amazing! 🚀

For issues: https://github.com/YOUR_USERNAME/ComfyUI-QwenVL-MultiImage/issues

