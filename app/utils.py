import tensorflow as tf
import numpy as np
import pickle
import os
import sys
from tensorflow.keras.utils import pad_sequences

tf.config.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"BASE_DIR: {BASE_DIR}")

# Load CPU-converted model
model_path = os.path.join(BASE_DIR, "models", "sentiment_model_cpu", "1")
if not os.path.exists(model_path):
    print(f"ERROR: Model path {model_path} does not exist!")
    sys.exit(1)

try:
    imported = tf.saved_model.load(model_path)
    infer = imported.signatures["serving_default"]
    print("✅ Model loaded successfully")
    print(f"Model input signature: {infer.structured_input_signature}")
    print(f"Model output keys: {list(infer.structured_outputs.keys())}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Dynamically determine the input key
if hasattr(infer, 'structured_input_signature'):
    sig = infer.structured_input_signature
    if sig and len(sig) > 1 and isinstance(sig[1], dict):
        input_keys = list(sig[1].keys())
        if input_keys:
            INPUT_KEY = input_keys[0]
            print(f"Detected input key: {INPUT_KEY}")
        else:
            INPUT_KEY = None
    else:
        INPUT_KEY = None
else:
    INPUT_KEY = None

# Fallback if detection fails
if INPUT_KEY is None:
    possible_keys = ['keras_tensor_72', 'keras_tensor_7', 'input_1', 'input', 'inputs']
    for key in possible_keys:
        try:
            dummy = tf.constant([[0]*30], dtype=tf.float32)
            infer(**{key: dummy})
            INPUT_KEY = key
            print(f"Using input key: {INPUT_KEY} (from fallback)")
            break
        except:
            continue
    else:
        print("❌ Could not determine input key from model signature.")
        sys.exit(1)

# Load tokenizer
tokenizer_path = os.path.join(BASE_DIR, "models", "tokenizer.pickle")
if not os.path.exists(tokenizer_path):
    print(f"ERROR: Tokenizer not found at {tokenizer_path}")
    sys.exit(1)

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
print("✅ Tokenizer loaded successfully")

# Test tokenizer
test_seq = tokenizer.texts_to_sequences(["test"])
print(f"Tokenizer test output: {test_seq}")

# Load label encoder
label_encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pickle")
if not os.path.exists(label_encoder_path):
    print(f"ERROR: Label encoder not found at {label_encoder_path}")
    sys.exit(1)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)
print("✅ Label encoder loaded successfully")
if hasattr(label_encoder, 'classes_'):
    print(f"Classes: {label_encoder.classes_}")

MAX_LEN = 30

def get_prediction(text):
    try:
        print(f"Received text for prediction: {text}", flush=True)
        seq = tokenizer.texts_to_sequences([text])
        print(f"Sequence: {seq}", flush=True)
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        print(f"Padded shape: {padded.shape}", flush=True)
        input_tensor = tf.convert_to_tensor(padded, dtype=tf.float32)
        print(f"Input tensor shape: {input_tensor.shape}", flush=True)

        # Call with dynamically determined input key
        preds_dict = infer(**{INPUT_KEY: input_tensor})
        print(f"Prediction dict keys: {list(preds_dict.keys())}", flush=True)

        if 'output_0' in preds_dict:
            predictions = preds_dict['output_0'].numpy()
        else:
            predictions = list(preds_dict.values())[0].numpy()

        print(f"Predictions shape: {predictions.shape}", flush=True)
        predicted_class = np.argmax(predictions, axis=1)
        print(f"Predicted class index: {predicted_class}", flush=True)
        label = label_encoder.inverse_transform(predicted_class)
        print(f"Predicted label: {label[0]}", flush=True)
        confidence = float(np.max(predictions))
        print(f"Confidence: {confidence}", flush=True)

        return {
            "sentiment": label[0],
            "confidence": confidence
        }
    except Exception as e:
        print(f"❌ Error in get_prediction: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise