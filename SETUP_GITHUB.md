# GitHub Setup Guide

This guide helps you publish your ComfyUI-QwenVL-MultiImage node to GitHub.

## Prerequisites

1. A GitHub account
2. Git installed on your system
3. Xcode Command Line Tools (macOS)

## Step 1: Fix Xcode License (macOS Only)

If you encountered an Xcode license error, run:

```bash
sudo xcodebuild -license
```

Press space to scroll through the license, then type `agree`.

## Step 2: Initialize Git Repository

```bash
cd /Users/hardik/Projects/comfyUI-qwenVL

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: ComfyUI QwenVL Multi-Image node"
```

## Step 3: Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the **+** icon in the top right
3. Select **New repository**
4. Fill in details:
   - **Repository name**: `ComfyUI-QwenVL-MultiImage`
   - **Description**: "ComfyUI custom node for Qwen2.5-VL and Qwen3-VL models with multi-image support"
   - **Public** (recommended for community use)
   - Do NOT initialize with README (we already have one)
5. Click **Create repository**

## Step 4: Push to GitHub

GitHub will show you commands. Use these:

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/ComfyUI-QwenVL-MultiImage.git

# Push to GitHub
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 5: Update README

Update the repository URL in these files:

1. `README.md`: Replace `YOUR_USERNAME` with your GitHub username
2. `config.json`: Update the repository URL

Then commit and push:

```bash
git add README.md config.json
git commit -m "Update repository URLs"
git push
```

## Step 6: Configure Repository

On GitHub, configure your repository:

### Add Topics

Click the gear icon next to "About" and add topics:
- `comfyui`
- `comfyui-custom-nodes`
- `qwen-vl`
- `qwen3-vl`
- `vision-language-model`
- `pytorch`
- `transformers`

### Create Releases

1. Click **Releases** → **Create a new release**
2. Tag version: `v1.0.0`
3. Release title: `v1.0.0 - Initial Release`
4. Description:
   ```
   Initial release of ComfyUI-QwenVL-MultiImage
   
   Features:
   - Multi-image support (up to 3 batches)
   - Qwen2.5-VL and Qwen3-VL model support
   - Standard and Advanced nodes
   - 4-bit, 8-bit, and FP16 quantization
   - Model caching for faster inference
   ```
5. Click **Publish release**

## Step 7: Add to ComfyUI Registry (Optional)

To make your node discoverable in ComfyUI Manager:

1. Fork the [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) repository
2. Edit `custom-node-list.json`
3. Add your node entry:
   ```json
   {
     "author": "YOUR_NAME",
     "title": "ComfyUI QwenVL Multi-Image",
     "reference": "https://github.com/YOUR_USERNAME/ComfyUI-QwenVL-MultiImage",
     "files": [
       "https://github.com/YOUR_USERNAME/ComfyUI-QwenVL-MultiImage"
     ],
     "install_type": "git-clone",
     "description": "Qwen2.5-VL and Qwen3-VL models with multi-image support for advanced vision-language AI"
   }
   ```
4. Submit a Pull Request

## Step 8: Share Your Work

Share your node with the community:

- Post on [ComfyUI Forum](https://comfyanonymous.github.io/ComfyUI_examples/)
- Share on Reddit r/comfyui
- Tweet about it with #ComfyUI
- Share in ComfyUI Discord

## Maintenance

### Making Updates

```bash
# Make your changes
git add .
git commit -m "Description of changes"
git push

# Create new release for major updates
# Use semantic versioning: v1.1.0, v1.2.0, v2.0.0, etc.
```

### Responding to Issues

- Enable issue notifications in GitHub settings
- Respond to issues within 48 hours if possible
- Label issues appropriately (bug, enhancement, question)
- Close resolved issues with explanations

## Need Help?

- Check [GitHub Docs](https://docs.github.com)
- Visit [ComfyUI Community](https://github.com/comfyanonymous/ComfyUI/discussions)
- Open an issue in this repository

Good luck with your node! 🚀

