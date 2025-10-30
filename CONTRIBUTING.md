# Contributing to ComfyUI-QwenVL-MultiImage

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ComfyUI-QwenVL-MultiImage.git
   cd ComfyUI-QwenVL-MultiImage
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Setup

### Testing Your Changes

1. Copy your development folder to ComfyUI's custom_nodes directory:
   ```bash
   ln -s /path/to/ComfyUI-QwenVL-MultiImage /path/to/ComfyUI/custom_nodes/
   ```

2. Restart ComfyUI and test your node

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to classes and functions
- Keep functions focused and modular

### Testing Guidelines

Before submitting a PR, please test:

1. **Single Image**: Verify node works with one image
2. **Multiple Images**: Test with 2-5 images
3. **Different Models**: Test at least 2 different model variants
4. **Quantization**: Test different quantization modes
5. **Error Handling**: Verify graceful error messages

## Submitting Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Screenshots/examples if applicable
   - Reference any related issues

## Feature Requests

Open an issue with:
- Clear description of the feature
- Use cases and benefits
- Any implementation ideas

## Bug Reports

Include:
- ComfyUI version
- Python version
- GPU/VRAM details
- Steps to reproduce
- Error messages/logs
- Screenshots if applicable

## Areas for Contribution

- Video input support
- Additional model integrations
- Performance optimizations
- Better error messages
- Documentation improvements
- Example workflows

## Questions?

Feel free to open an issue for any questions!

Thank you for contributing! 🙏

