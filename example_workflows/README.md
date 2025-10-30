# Example Workflows

This directory contains example ComfyUI workflows demonstrating the QwenVL Multi-Image node capabilities.

## Available Workflows

### 1. basic_multi_image.json
**Description**: Simple two-image comparison workflow
- Loads 2 images
- Uses standard QwenVL_MultiImage node
- Compares and describes similarities/differences
- Good starting point for new users

**Use Case**: Product comparison, before/after analysis, style comparison

### 2. advanced_multi_image.json
**Description**: Advanced three-image analysis with full parameter control
- Loads 3 images
- Uses QwenVL_MultiImage_Advanced node
- Full control over temperature, top_p, repetition penalty
- Demonstrates all optional image inputs

**Use Case**: Sequential analysis, multi-view understanding, detailed comparisons

## How to Use

1. Open ComfyUI
2. Click **Load** in the menu
3. Navigate to this directory
4. Select a workflow JSON file
5. Replace the example images with your own
6. Adjust prompts and parameters as needed
7. Run the workflow

## Creating Your Own Workflows

### Single Image Analysis
```
LoadImage → QwenVL_MultiImage → ShowText
```

### Multiple Image Comparison
```
LoadImage (x2 or x3) → QwenVL_MultiImage → ShowText
```

### Batch Processing
```
LoadImageBatch → QwenVL_MultiImage → ShowText/SaveText
```

### Chained Analysis
```
LoadImage → QwenVL_MultiImage → [Process Text] → Another Node
```

## Tips

1. **Image Order Matters**: If you want to reference specific images, number them in your prompt ("In the first image...", "Comparing image 1 and 2...")

2. **System Prompts**: Use system prompts to set the context:
   - "You are an expert photographer analyzing composition"
   - "You are a medical imaging specialist"
   - "You are a product analyst comparing features"

3. **User Prompts**: Be specific:
   - ❌ "Describe these images"
   - ✅ "Compare the color schemes, composition, and mood of these three images"

4. **Performance**: 
   - Start with smaller models (2B, 3B, 4B) for testing
   - Use 8-bit quantization as default
   - Enable `keep_model_loaded` for multiple runs

## Community Workflows

Have an interesting workflow? Share it with the community!

1. Export your workflow from ComfyUI
2. Add it to this directory
3. Update this README with a description
4. Submit a Pull Request

## Need Help?

- Check the main [README.md](../README.md) for detailed parameter explanations
- Open an issue on GitHub for workflow-specific questions
- Share your workflows in ComfyUI Discord/Reddit communities

