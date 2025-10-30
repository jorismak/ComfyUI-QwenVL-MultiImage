#!/usr/bin/env python3
"""
Test script to verify ComfyUI-QwenVL-MultiImage installation
Run this from the ComfyUI/custom_nodes/ComfyUI-QwenVL-MultiImage directory
"""

import sys
import importlib.util

def test_imports():
    """Test if all required packages are importable"""
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'accelerate': 'Accelerate',
        'bitsandbytes': 'BitsAndBytes',
    }
    
    print("Testing required package imports...")
    all_ok = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} ({package})")
        except ImportError as e:
            print(f"✗ {name} ({package}) - NOT FOUND")
            print(f"  Error: {e}")
            all_ok = False
    
    # Test qwen_vl_utils separately as it might not be installed
    try:
        import qwen_vl_utils
        print(f"✓ qwen-vl-utils")
    except ImportError:
        print(f"✗ qwen-vl-utils - NOT FOUND (will be needed at runtime)")
        all_ok = False
    
    return all_ok

def test_torch_cuda():
    """Test PyTorch CUDA availability"""
    print("\nTesting PyTorch CUDA setup...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
        else:
            print("⚠ CUDA not available - will run on CPU (much slower)")
        return True
    except Exception as e:
        print(f"✗ Error testing PyTorch: {e}")
        return False

def test_node_import():
    """Test if the node module can be imported"""
    print("\nTesting node import...")
    try:
        from nodes import QwenVL_MultiImage, QwenVL_MultiImage_Advanced
        print("✓ QwenVL_MultiImage node class")
        print("✓ QwenVL_MultiImage_Advanced node class")
        
        # Test that classes have required methods
        assert hasattr(QwenVL_MultiImage, 'INPUT_TYPES')
        assert hasattr(QwenVL_MultiImage, 'generate')
        assert hasattr(QwenVL_MultiImage_Advanced, 'INPUT_TYPES')
        assert hasattr(QwenVL_MultiImage_Advanced, 'generate')
        print("✓ Node classes have required methods")
        
        return True
    except Exception as e:
        print(f"✗ Error importing nodes: {e}")
        return False

def test_transformers_version():
    """Check if transformers version is sufficient"""
    print("\nChecking transformers version...")
    try:
        import transformers
        version = transformers.__version__
        print(f"Transformers version: {version}")
        
        # Parse version
        major, minor = map(int, version.split('.')[:2])
        if major >= 4 and minor >= 45:
            print("✓ Transformers version is sufficient")
            return True
        else:
            print(f"⚠ Transformers version {version} may be too old (need >=4.45.0)")
            return False
    except Exception as e:
        print(f"✗ Error checking transformers: {e}")
        return False

def main():
    print("=" * 60)
    print("ComfyUI-QwenVL-MultiImage Installation Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("PyTorch CUDA", test_torch_cuda),
        ("Transformers Version", test_transformers_version),
        ("Node Import", test_node_import),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Installation looks good.")
        print("\nNext steps:")
        print("1. Restart ComfyUI")
        print("2. Look for '🧪 QwenVL Multi-Image' in the node menu")
        print("3. Try the example workflows in example_workflows/")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Run: pip install -r requirements.txt")
        print("- Install CUDA toolkit for GPU support")
        print("- Update transformers: pip install --upgrade transformers")
        return 1

if __name__ == "__main__":
    sys.exit(main())

