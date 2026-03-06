import tensorflow as tf
import numpy as np
import pickle
import os
import sys
from tensorflow.keras.utils import pad_sequences
import json

tf.config.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== MODEL CONFIGURATION ====================
AVAILABLE_MODELS = {
    "version1": {
        "model_path": os.path.join(BASE_DIR, "models", "version1", "sentiment_model", "1"),
        "tokenizer_path": os.path.join(BASE_DIR, "models", "version1", "tokenizer.pickle"),
        "encoder_path": os.path.join(BASE_DIR, "models", "version1", "label_encoder.pickle"),
        "metadata_path": os.path.join(BASE_DIR, "models", "version1", "metadata.json"),
        "max_len": 30
    },
    "version2": {
        "model_path": os.path.join(BASE_DIR, "models", "version2", "sentiment_model", "1"),
        "tokenizer_path": os.path.join(BASE_DIR, "models", "version2", "tokenizer.pickle"),
        "encoder_path": os.path.join(BASE_DIR, "models", "version2", "label_encoder.pickle"),
        "metadata_path": os.path.join(BASE_DIR, "models", "version2", "metadata.json"),
        "max_len": 30
    }
}

# ==================== LOAD ALL MODELS ====================
models = {}
tokenizers = {}
label_encoders = {}
input_keys = {}
metadata = {}

print("="*60)
print("LOADING MODEL VERSIONS")
print("="*60)

for version, config in AVAILABLE_MODELS.items():
    print(f"\nLoading {version}...")
    
    # Check if model exists
    if not os.path.exists(config["model_path"]):
        print(f"  ⚠ {version} model not found at {config['model_path']}")
        print(f"  ℹ Skipping {version}")
        continue
    
    try:
        # Load model
        imported = tf.saved_model.load(config["model_path"])
        infer = imported.signatures["serving_default"]
        models[version] = infer
        
        # Detect input key
        if hasattr(infer, 'structured_input_signature'):
            sig = infer.structured_input_signature
            if sig and len(sig) > 1 and isinstance(sig[1], dict):
                input_keys_list = list(sig[1].keys())
                if input_keys_list:
                    input_keys[version] = input_keys_list[0]
        
        # Load tokenizer
        with open(config["tokenizer_path"], 'rb') as f:
            tokenizers[version] = pickle.load(f)
        
        # Load label encoder
        with open(config["encoder_path"], 'rb') as f:
            label_encoders[version] = pickle.load(f)
        
        # Load metadata
        if os.path.exists(config["metadata_path"]):
            with open(config["metadata_path"], 'r') as f:
                metadata[version] = json.load(f)
        else:
            metadata[version] = {"version": version, "description": "No metadata available"}
        
        print(f"  ✅ {version} loaded successfully")
        print(f"     Model: {os.path.basename(config['model_path'])}")
        print(f"     Input key: {input_keys.get(version, 'N/A')}")
        if version in metadata:
            print(f"     Description: {metadata[version].get('description', 'N/A')}")
        
    except Exception as e:
        print(f"  ❌ Failed to load {version}: {e}")

print("\n" + "="*60)
print(f"LOADED MODELS: {list(models.keys())}")
print("="*60)

if not models:
    print("ERROR: No models loaded! Please run setup_model_versions.py and train_model_v2.py")
    sys.exit(1)

# ==================== PREDICTION FUNCTION ====================
def get_prediction(text, model_version="version1"):
    """
    Get sentiment prediction from specified model version
    
    Args:
        text: Input text to analyze
        model_version: Which model to use ("version1" or "version2")
    
    Returns:
        dict: Contains sentiment, confidence, and model info
    """
    try:
        # Validate model version
        if model_version not in models:
            available = list(models.keys())
            print(f"Warning: {model_version} not available. Using {available[0]}")
            model_version = available[0]
        
        config = AVAILABLE_MODELS[model_version]
        model = models[model_version]
        tokenizer = tokenizers[model_version]
        label_encoder = label_encoders[model_version]
        input_key = input_keys[model_version]
        max_len = config["max_len"]
        
        # Preprocess
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        input_tensor = tf.convert_to_tensor(padded, dtype=tf.float32)
        
        # Predict
        preds_dict = model(**{input_key: input_tensor})
        output_key = list(preds_dict.keys())[0]
        predictions = preds_dict[output_key].numpy()
        
        # Get result
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_label = label_encoder.classes_[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        return {
            "sentiment": predicted_label,
            "confidence": confidence,
            "model_version": model_version,
            "model_info": metadata.get(model_version, {})
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise

def get_available_models():
    """Get list of available model versions with metadata"""
    return {
        version: {
            "available": True,
            "metadata": metadata.get(version, {})
        }
        for version in models.keys()
    }
