import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import tensorflow as tf

st.set_page_config(page_title="IMDB Movie Review Classifier", page_icon="🎬")
st.title("IMDB Movie Review Classifier by Anu")

# Load IMDB dataset
(_, _), (xtest, ytest) = imdb.load_data(num_words=10000)

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for (key, value) in word_index.items()}
reverse_word_index[0] = '<PAD>'
reverse_word_index[1] = '<START>'
reverse_word_index[2] = '<UNK>'
reverse_word_index[3] = 'the'

def decode_review(seq):
    return ' '.join([reverse_word_index.get(i, '?') for i in seq if i != 0])

# Try loading model, but don’t break Streamlit if fails
try:
    lstm_model = tf.keras.models.load_model("lstm_imdb.h5")
    model_loaded = True
except Exception as e:
    st.warning(f"Could not load model: {e}")
    lstm_model = None
    model_loaded = False

st.header("5 Sample Test Reviews")
for i in range(5):
    seq = xtest[i]
    text = decode_review(seq)

    st.subheader(f"Sample {i+1}")
    st.write("Review (truncated):", text[:600])
    
    actual = 'Positive' if ytest[i] == 1 else 'Negative'
    st.write("Actual:", actual)

    # Predict if model loaded
    if model_loaded:
        seq_padded = pad_sequences([seq], maxlen=200, padding='post')
        seq_tensor = tf.convert_to_tensor(seq_padded)
        prob = float(lstm_model.predict(seq_tensor, verbose=0)[0][0])
        pred = 'Positive' if prob >= 0.5 else 'Negative'
        st.write(f"Predicted: {pred} (prob={prob:.4f})")
    else:
        st.write("Predicted: Model not loaded")

    st.markdown("---")
