import os
import shutil
import json

print("="*60)
print("SETTING UP MODEL VERSIONS")
print("="*60)

# Create version directories
os.makedirs("models/version1", exist_ok=True)
os.makedirs("models/version2", exist_ok=True)

# ==================== VERSION 1 - Copy existing model ====================
print("\n[1/2] Setting up Version 1 (Current Model)...")

# Copy existing model to version1
src_model = "models/sentiment_model_cpu/1"
dst_model = "models/version1/sentiment_model/1"

if os.path.exists(src_model):
    if os.path.exists(dst_model):
        shutil.rmtree(dst_model)
    shutil.copytree(src_model, dst_model)
    print(f"  ✓ Model copied to {dst_model}")
else:
    print(f"  ⚠ Warning: Original model not found at {src_model}")
    print("    Please run train_cpu_model.py first!")

# Copy tokenizer
if os.path.exists("models/tokenizer.pickle"):
    shutil.copy("models/tokenizer.pickle", "models/version1/tokenizer.pickle")
    print("  ✓ Tokenizer copied")

# Copy label encoder
if os.path.exists("models/label_encoder.pickle"):
    shutil.copy("models/label_encoder.pickle", "models/version1/label_encoder.pickle")
    print("  ✓ Label encoder copied")

# Create metadata for version1
metadata_v1 = {
    "version": "1",
    "description": "Original baseline model",
    "vocab_size": 500,
    "embedding_dim": 64,
    "epochs": 4,
    "architecture": "Bi-LSTM single layer"
}

with open("models/version1/metadata.json", "w") as f:
    json.dump(metadata_v1, f, indent=2)
print("  ✓ Metadata created")

# ==================== VERSION 2 ====================
print("\n[2/2] Version 2 Setup...")
print("  ℹ Version 2 will be created when you run: python train_model_v2.py")

# Create placeholder metadata for version2
metadata_v2 = {
    "version": "2",
    "description": "Enhanced model with deeper architecture",
    "vocab_size": 1000,
    "embedding_dim": 128,
    "epochs": 10,
    "architecture": "Bi-LSTM x 2 layers",
    "status": "Not trained yet - Run train_model_v2.py"
}

with open("models/version2/metadata.json", "w") as f:
    json.dump(metadata_v2, f, indent=2)

print("\n" + "="*60)
print("SETUP COMPLETE!")
print("="*60)
print("\nModel Versions:")
print("  📁 version1/ - Current model (baseline)")
print("  📁 version2/ - Enhanced model (run train_model_v2.py)")
print("\nNext steps:")
print("  1. Run: python train_model_v2.py")
print("  2. Both models will be available in the application")
print("="*60)
