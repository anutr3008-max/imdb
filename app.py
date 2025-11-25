import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import tensorflow as tf
import os

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="IMDB Movie Review Classifier", page_icon="🎬")
st.title("IMDB Movie Review Classifier by Anu")


# ---------------------------
# Load IMDB dataset
# ---------------------------
(_, _), (xtest, ytest) = imdb.load_data(num_words=10000)

# ---------------------------
# Load IMDB word index
# ---------------------------
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for (key, value) in word_index.items()}
reverse_word_index[0] = '<PAD>'
reverse_word_index[1] = '<START>'
reverse_word_index[2] = '<UNK>'
reverse_word_index[3] = 'the'

def decode_review(seq):
    """Convert sequence of integers back to words"""
    return ' '.join([reverse_word_index.get(i, '?') for i in seq if i != 0])

# ---------------------------
# Load LSTM model
# ---------------------------
model_path = "lstm_imdb_savedmodel"

if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check the path.")
    lstm_model = None
    model_loaded = False
else:
    try:
        lstm_model = tf.keras.models.load_model(model_path)
        model_loaded = True
        st.success(f"Loaded model from {model_path}")
    except Exception as e:
        st.error(f"Could not load model:\n{e}")
        lstm_model = None
        model_loaded = False

# ---------------------------
# Display 5 sample test reviews
# ---------------------------
st.header("5 Sample Test Reviews")
for i in range(5):
    seq = xtest[i]
    text = decode_review(seq)

    st.subheader(f"Sample {i+1}")
    st.write("Review (truncated):", text[:600])
    
    actual = 'Positive' if ytest[i] == 1 else 'Negative'
    st.write("Actual:", actual)

    if model_loaded:
        seq_padded = pad_sequences([seq], maxlen=500, padding='post')
        seq_tensor = tf.convert_to_tensor(seq_padded)
        try:
            prob = float(lstm_model.predict(seq_tensor, verbose=0)[0][0])
            pred = 'Positive' if prob >= 0.5 else 'Negative'
            st.write(f"Predicted: {pred} (prob={prob:.4f})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.write("Predicted: Model not loaded")

    st.markdown("---")

# ---------------------------
# Optional: Classify custom review
# ---------------------------
st.header("Classify Your Own Review")
user_input = st.text_area("Type your IMDB review here:")

if st.button("Predict Review Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review to predict.")
    elif not model_loaded:
        st.error("LSTM model not loaded.")
    else:
        # Encode review using IMDB word index
        words = user_input.lower().split()
        seq = [word_index.get(word, 2) for word in words]  # 2 = <UNK>
        seq_padded = pad_sequences([seq], maxlen=500, padding='post')
        seq_tensor = tf.convert_to_tensor(seq_padded)
        try:
            prob = float(lstm_model.predict(seq_tensor, verbose=0)[0][0])
            pred = "Positive" if prob >= 0.5 else "Negative"
            st.success(f"Predicted: {pred} (prob={prob:.4f})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


