import os
import sys
import pickle
import json
import numpy as np

# Set TensorFlow configuration BEFORE importing it to optimize memory
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Memory Optimization: Disable GPU & limit thread usage
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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

# Caches for lazy loaded components
_models = {}
_tokenizers = {}
_label_encoders = {}
_input_keys = {}
_metadata = {}

def pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    """Custom pad_sequences implementation to avoid importing heavy Keras modules"""
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:
                seq = seq[:maxlen]
        else:
            difference = maxlen - len(seq)
            if padding == 'pre':
                seq = [value] * difference + seq
            else:
                seq = seq + [value] * difference
        padded.append(seq)
    return np.array(padded)

def load_model_version(version):
    """Lazily load a specific model version when requested"""
    global _models, _tokenizers, _label_encoders, _input_keys, _metadata
    
    if version in _models:
        return True
        
    if version not in AVAILABLE_MODELS:
        print(f"  ❌ Requested model version '{version}' is not defined.")
        return False
        
    config = AVAILABLE_MODELS[version]
    
    if not os.path.exists(config["model_path"]):
        print(f"  ⚠ {version} model not found at {config['model_path']}")
        return False
        
    try:
        print(f"⏳ Loading {version} lazily...", flush=True)
        # Load model signature
        imported = tf.saved_model.load(config["model_path"])
        infer = imported.signatures["serving_default"]
        _models[version] = infer
        
        # Detect input key
        if hasattr(infer, 'structured_input_signature'):
            sig = infer.structured_input_signature
            if sig and len(sig) > 1 and isinstance(sig[1], dict):
                input_keys_list = list(sig[1].keys())
                if input_keys_list:
                    _input_keys[version] = input_keys_list[0]
                    
        if version not in _input_keys:
            # Fallback
            _input_keys[version] = "input_1"
            
        # Load tokenizer
        with open(config["tokenizer_path"], 'rb') as f:
            _tokenizers[version] = pickle.load(f)
            
        # Load label encoder
        with open(config["encoder_path"], 'rb') as f:
            _label_encoders[version] = pickle.load(f)
            
        # Load metadata
        if os.path.exists(config["metadata_path"]):
            with open(config["metadata_path"], 'r') as f:
                _metadata[version] = json.load(f)
        else:
            _metadata[version] = {"version": version, "description": f"Dynamic metadata for {version}"}
            
        print(f"  ✅ {version} loaded successfully!", flush=True)
        return True
    except Exception as e:
        print(f"  ❌ Failed to load {version}: {e}", flush=True)
        return False

def get_prediction(text, model_version="version1"):
    try:
        # Lazily load the target model version
        success = load_model_version(model_version)
        if not success:
            # Fallback to any loaded model
            if _models:
                fallback_version = list(_models.keys())[0]
                print(f"Warning: {model_version} failed to load. Falling back to {fallback_version}")
                model_version = fallback_version
            else:
                # Try to load the other configured one as fallback
                other_versions = [v for v in AVAILABLE_MODELS.keys() if v != model_version]
                for other in other_versions:
                    if load_model_version(other):
                        model_version = other
                        break
                else:
                    raise RuntimeError("No models could be loaded.")
                    
        config = AVAILABLE_MODELS[model_version]
        model = _models[model_version]
        tokenizer = _tokenizers[model_version]
        label_encoder = _label_encoders[model_version]
        input_key = _input_keys[model_version]
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
            "model_info": _metadata.get(model_version, {})
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise

def get_available_models():
    # Attempt to load metadata for all configured models
    available = {}
    for version in AVAILABLE_MODELS.keys():
        # Check if the directories exist physically
        if os.path.exists(AVAILABLE_MODELS[version]["model_path"]):
            # Load metadata (lightweight file read, no model loading)
            metadata_path = AVAILABLE_MODELS[version]["metadata_path"]
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                except:
                    meta = {"version": version, "description": f"Model version {version}"}
            else:
                meta = {"version": version, "description": f"Model version {version}"}
            available[version] = {
                "available": True,
                "metadata": meta
            }
    return available
