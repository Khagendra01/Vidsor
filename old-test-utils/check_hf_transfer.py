"""
Check if HuggingFace Rust-based fast downloader (hf_transfer) is enabled and working.
"""

import os
import sys


def check_hf_transfer():
    """Check hf_transfer status and provide installation instructions."""
    print("="*60)
    print("HuggingFace Transfer (hf_transfer) Status Check")
    print("="*60)
    
    # Check environment variable
    env_var = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
    print(f"\n1. Environment Variable:")
    print(f"   HF_HUB_ENABLE_HF_TRANSFER = {env_var}")
    
    if env_var == "1":
        print("   ✅ Fast downloader is ENABLED")
    else:
        print("   ❌ Fast downloader is DISABLED")
        print("   💡 Enable it with: set HF_HUB_ENABLE_HF_TRANSFER=1")
    
    # Check if hf_transfer package is installed
    print(f"\n2. Package Installation:")
    try:
        import hf_transfer
        print("   ✅ hf_transfer package is INSTALLED")
        try:
            print(f"   Version: {hf_transfer.__version__}")
        except:
            print("   Version: (unknown)")
    except ImportError:
        print("   ❌ hf_transfer package is NOT INSTALLED")
        print("   💡 Install it with: pip install hf_transfer")
    
    # Check if it's actually being used
    print(f"\n3. Usage Check:")
    try:
        from huggingface_hub import file_download
        # Check if hf_transfer is available in huggingface_hub
        if hasattr(file_download, '_hf_transfer_available'):
            print("   ✅ HuggingFace Hub can use hf_transfer")
        else:
            print("   ⚠️  Cannot verify if hf_transfer is available to HuggingFace Hub")
    except Exception as e:
        print(f"   ⚠️  Could not check: {e}")
    
    # Instructions
    print("\n" + "="*60)
    print("How to Enable Fast Downloads")
    print("="*60)
    print("\nOption 1: Set environment variable (Windows PowerShell):")
    print("   $env:HF_HUB_ENABLE_HF_TRANSFER='1'")
    print("\nOption 2: Set environment variable (Windows CMD):")
    print("   set HF_HUB_ENABLE_HF_TRANSFER=1")
    print("\nOption 3: Install hf_transfer package:")
    print("   pip install hf_transfer")
    print("\nOption 4: Set in Python code (before importing transformers):")
    print("   import os")
    print("   os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'")
    
    print("\n" + "="*60)
    print("Expected Speed Improvements")
    print("="*60)
    print("   Python downloader:  ~50-200 kB/s (slow)")
    print("   Rust (hf_transfer): ~5-50 MB/s (10-50x faster)")
    print("\n   For large models like LLaVA (15GB+), this can save hours!")
    
    print("\n" + "="*60)
    print("Troubleshooting")
    print("="*60)
    print("   If downloads still seem slow:")
    print("   1. Make sure hf_transfer is installed: pip install hf_transfer")
    print("   2. Check your internet connection speed")
    print("   3. Try disabling if causing errors: set HF_HUB_ENABLE_HF_TRANSFER=0")
    print("   4. Check HuggingFace server status")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    check_hf_transfer()

