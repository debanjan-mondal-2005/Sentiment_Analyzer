import tensorflow as tf
import pandas as pd
import pickle
import os
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("TRAINING MODEL VERSION 2 - Enhanced Architecture")
print("="*60)

tf.config.set_visible_devices([], 'GPU')
print("Using CPU only")

MAX_VOCAB_SIZE = 1000  
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 128  
BATCH_SIZE = 32
EPOCHS = 10 

print(f"\nVersion 2 Parameters:")
print(f"  - Vocab Size: {MAX_VOCAB_SIZE}")
print(f"  - Embedding Dim: {EMBEDDING_DIM}")
print(f"  - Epochs: {EPOCHS}")

print("\nLoading data...")
train_df = pd.read_csv("data/social_media_sentiment_train.csv")
test_df = pd.read_csv("data/social_media_sentiment_test.csv")

text_col = 'text'
label_col = 'label'

print("Creating tokenizer...")
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df[text_col])

os.makedirs("models/version2", exist_ok=True)
with open("models/version2/tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

X_train = tokenizer.texts_to_sequences(train_df[text_col])
X_test = tokenizer.texts_to_sequences(test_df[text_col])
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")


print("Encoding labels...")
encoder = LabelEncoder()
y_train = encoder.fit_transform(train_df[label_col])
y_test = encoder.transform(test_df[label_col])

with open("models/version2/label_encoder.pickle", "wb") as f:
    pickle.dump(encoder, f)

num_classes = len(encoder.classes_)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

print("\nBuilding Version 2 model (Enhanced Architecture)...")
model = Sequential([
    Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH
    ),
    Bidirectional(LSTM(128, return_sequences=True, implementation=1)),  
    Dropout(0.3),
    Bidirectional(LSTM(64, implementation=1)), 
    Dropout(0.5),
    Dense(64, activation="relu"), 
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print(model.summary())
print("\nTraining Version 2...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

save_path = os.path.join("models", "version2", "sentiment_model", "1")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.export(save_path)
print(f"\nVersion 2 Model saved to {save_path}")

metadata = {
    "version": "2",
    "description": "Enhanced model with deeper architecture",
    "vocab_size": MAX_VOCAB_SIZE,
    "embedding_dim": EMBEDDING_DIM,
    "epochs": EPOCHS,
    "architecture": "Bi-LSTM x 2 layers",
    "final_accuracy": float(history.history['accuracy'][-1]),
    "final_val_accuracy": float(history.history['val_accuracy'][-1])
}

import json
with open("models/version2/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*60)
print("VERSION 2 TRAINING COMPLETE!")
print("="*60)
print(f"Training Accuracy: {metadata['final_accuracy']:.4f}")
print(f"Validation Accuracy: {metadata['final_val_accuracy']:.4f}")
print("="*60)
