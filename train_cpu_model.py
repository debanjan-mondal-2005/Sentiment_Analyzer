import tensorflow as tf
import pandas as pd
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

tf.config.set_visible_devices([], 'GPU')
print("Using CPU only")

MAX_VOCAB_SIZE = 500
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 4

train_df = pd.read_csv("data/social_media_sentiment_train.csv")
test_df = pd.read_csv("data/social_media_sentiment_test.csv")

text_col = 'text'
label_col = 'label'

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df[text_col])

with open("models/tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

X_train = tokenizer.texts_to_sequences(train_df[text_col])
X_test = tokenizer.texts_to_sequences(test_df[text_col])
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

encoder = LabelEncoder()
y_train = encoder.fit_transform(train_df[label_col])
y_test = encoder.transform(test_df[label_col])

with open("models/label_encoder.pickle", "wb") as f:
    pickle.dump(encoder, f)

num_classes = len(encoder.classes_)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

model = Sequential([
    Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH
    ),
    Bidirectional(LSTM(64, implementation=1)), 
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print(model.summary())

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)
save_path = os.path.join("models", "sentiment_model_cpu", "1")
model.export(save_path) 
print(f"Model saved to {save_path}")

imported = tf.saved_model.load(save_path)
infer = imported.signatures['serving_default']
print("Inputs:", infer.structured_input_signature)
print("Outputs:", infer.structured_outputs)
print("Model ready for CPU inference.")