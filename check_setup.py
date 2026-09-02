#!/usr/bin/env python3
"""
Setup verification script for People Counter (Offline Mode)
Run this to check if everything is ready.
"""

import sys
from pathlib import Path

def check():
    print("=" * 60)
    print("  PEOPLE COUNTER - SETUP VERIFICATION")
    print("=" * 60)

    # Check Python version
    print(f"✓ Python: {sys.version.split()[0]}")

    # Check required packages
    packages = ["fastapi", "uvicorn", "cv2", "numpy", "ultralytics"]
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg:15} installed")
        except ImportError:
            print(f"✗ {pkg:15} MISSING - run: pip install {pkg}")
            missing.append(pkg)

    # Check model files
    print("" + "-" * 60)
    print("MODEL FILES:")
    models = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", 
              "yolov8l-seg.pt", "yolov8x-seg.pt"]

    found_any = False
    for model in models:
        if Path(model).exists():
            size = Path(model).stat().st_size / (1024*1024)
            print(f"✓ {model:20} FOUND ({size:.1f} MB)")
            found_any = True
        else:
            print(f"✗ {model:20} NOT FOUND")

    # Check ultralytics cache
    cache_dir = Path.home() / ".ultralytics" / "weights"
    if cache_dir.exists():
        for model in models:
            if (cache_dir / model).exists():
                size = (cache_dir / model).stat().st_size / (1024*1024)
                print(f"✓ {model:20} FOUND in cache ({size:.1f} MB)")
                found_any = True

    if not found_any:
        print("⚠️  NO MODEL FILES FOUND!")
        print("   You need to download a model file manually.")
        print("   See README.md or run: python download_models.py")

    print("" + "=" * 60)

    if missing:
        print("❌ Setup incomplete. Install missing packages first.")
        sys.exit(1)
    elif not found_any:
        print("⚠️  Packages OK, but model file missing.")
        sys.exit(2)
    else:
        print("✅ All checks passed! Run: python app.py")
        sys.exit(0)

if __name__ == "__main__":
    check()
