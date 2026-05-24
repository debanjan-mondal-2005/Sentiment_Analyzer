import os
import sys
import pickle
import numpy as np

# Set TensorFlow configuration BEFORE importing it to optimize memory
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress info/warning logs

import tensorflow as tf

# Memory Optimization: Disable GPU & limit thread usage
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global cache for lazy loading
_model = None
_infer = None
_tokenizer = None
_label_encoder = None
_INPUT_KEY = None

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

def load_resources():
    global _model, _infer, _tokenizer, _label_encoder, _INPUT_KEY
    
    if _model is not None:
        return
        
    model_path = os.path.join(BASE_DIR, "models", "sentiment_model_cpu", "1")
    if not os.path.exists(model_path):
        print(f"ERROR: Model path {model_path} does not exist!")
        sys.exit(1)
        
    try:
        print("⏳ Loading CPU model (lazily)...", flush=True)
        _model = tf.saved_model.load(model_path)
        _infer = _model.signatures["serving_default"]
        print("✅ Model loaded successfully", flush=True)
    except Exception as e:
        print(f"❌ Failed to load model: {e}", flush=True)
        sys.exit(1)
        
    # Dynamically determine the input key
    if hasattr(_infer, 'structured_input_signature'):
        sig = _infer.structured_input_signature
        if sig and len(sig) > 1 and isinstance(sig[1], dict):
            input_keys = list(sig[1].keys())
            if input_keys:
                _INPUT_KEY = input_keys[0]
                print(f"Detected input key: {_INPUT_KEY}", flush=True)
                
    if _INPUT_KEY is None:
        possible_keys = ['keras_tensor_72', 'keras_tensor_7', 'input_1', 'input', 'inputs']
        for key in possible_keys:
            try:
                dummy = tf.constant([[0]*30], dtype=tf.float32)
                _infer(**{key: dummy})
                _INPUT_KEY = key
                print(f"Using input key: {_INPUT_KEY} (from fallback)", flush=True)
                break
            except:
                continue
        else:
            print("❌ Could not determine input key from model signature.", flush=True)
            sys.exit(1)
            
    # Load tokenizer
    tokenizer_path = os.path.join(BASE_DIR, "models", "tokenizer.pickle")
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}", flush=True)
        sys.exit(1)
        
    with open(tokenizer_path, 'rb') as f:
        _tokenizer = pickle.load(f)
    print("✅ Tokenizer loaded successfully", flush=True)
    
    # Load label encoder
    label_encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pickle")
    if not os.path.exists(label_encoder_path):
        print(f"ERROR: Label encoder not found at {label_encoder_path}", flush=True)
        sys.exit(1)
        
    with open(label_encoder_path, 'rb') as f:
        _label_encoder = pickle.load(f)
    print("✅ Label encoder loaded successfully", flush=True)

def get_prediction(text, model_version=None):
    try:
        load_resources()
        print(f"Received text for prediction: {text}", flush=True)
        seq = _tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=30, padding="post", truncating="post")
        input_tensor = tf.convert_to_tensor(padded, dtype=tf.float32)
        
        preds_dict = _infer(**{_INPUT_KEY: input_tensor})
        
        if 'output_0' in preds_dict:
            predictions = preds_dict['output_0'].numpy()
        else:
            predictions = list(preds_dict.values())[0].numpy()
            
        predicted_class = np.argmax(predictions, axis=1)
        label = _label_encoder.inverse_transform(predicted_class)
        confidence = float(np.max(predictions))
        
        return {
            "sentiment": label[0],
            "confidence": confidence
        }
    except Exception as e:
        print(f"❌ Error in get_prediction: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise