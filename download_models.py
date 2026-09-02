#!/usr/bin/env python3
"""
Model Download Helper for People Counter
==========================================
This script helps you download YOLO segmentation models.

If you're offline, use a device WITH internet to download from:
https://github.com/ultralytics/assets/releases/

Then copy the .pt file to this folder.
"""

import urllib.request
import sys
from pathlib import Path

MODELS = {
    "1": ("yolov8n-seg.pt", "6 MB", "Fastest, lowest accuracy"),
    "2": ("yolov8s-seg.pt", "23 MB", "Fast, good accuracy"),
    "3": ("yolov8m-seg.pt", "68 MB", "Balanced"),
    "4": ("yolov8l-seg.pt", "147 MB", "High accuracy"),
    "5": ("yolov8x-seg.pt", "248 MB", "Best accuracy"),
}

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"

def download_model(choice):
    model_name, size, desc = MODELS[choice]
    url = BASE_URL + model_name

    print(f"\nDownloading {model_name} ({size}) - {desc}")
    print(f"URL: {url}")
    print("This may take a few minutes...\n")

    try:
        urllib.request.urlretrieve(url, model_name)
        print(f"✅ Successfully downloaded: {model_name}")
        print(f"   Location: {Path(model_name).absolute()}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nYou appear to be OFFLINE.")
        print("Please download manually from a device with internet:")
        print(f"   {url}")
        print(f"Then copy the file to: {Path.cwd()}")
        return False

def main():
    print("=" * 60)
    print("  YOLO SEGMENTATION MODEL DOWNLOADER")
    print("=" * 60)
    print("\nSelect a model to download:")

    for key, (name, size, desc) in MODELS.items():
        print(f"  [{key}] {name:18} {size:>6}  - {desc}")

    print("  [0] Exit")
    print()

    choice = input("Enter choice (1-5): ").strip()

    if choice == "0":
        sys.exit(0)

    if choice not in MODELS:
        print("Invalid choice.")
        sys.exit(1)

    success = download_model(choice)

    if success:
        print("\n✅ You can now run: python app.py")
    else:
        print("\n⚠️  Manual download required.")

if __name__ == "__main__":
    main()
