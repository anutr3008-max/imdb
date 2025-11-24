import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# ----------------------
# Load LSTM SavedModel
# ----------------------
lstm_model_path = "lstm_imdb_savedmodel"
lstm_model = tf.saved_model.load(lstm_model_path)
serve_fn = lstm_model.signatures["serving_default"]

# ----------------------
# Load IMDB word index and decoder
# ----------------------
word_index = imdb.get_word_index()
reverse_word_index = {value+3: key for (key, value) in word_index.items()}
reverse_word_index[0] = '<PAD>'
reverse_word_index[1] = '<START>'
reverse_word_index[2] = '<UNK>'
reverse_word_index[3] = 'the'

def decode_review(seq):
    return ' '.join([reverse_word_index.get(i, '?') for i in seq if i != 0])

def encode_review(text, maxlen=200):
    words = text.lower().split()
    seq = [word_index[word]+3 for word in words if word in word_index]
    return pad_sequences([seq], maxlen=maxlen, padding='post')

# ----------------------
# Helper function for prediction
# ----------------------
def predict_prob(seq):
    seq_padded = pad_sequences([seq], maxlen=200, padding='post')
    seq_tensor = tf.convert_to_tensor(seq_padded, dtype=tf.int32)

    # Get input name from SavedModel signature
    input_name = list(serve_fn.structured_input_signature[1].keys())[0]

    # Call the model
    output = serve_fn(**{input_name: seq_tensor})

    # Get the first output tensor value
    prob = list(output.values())[0].numpy()[0,0]
    return prob

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="IMDB Movie Review Classifier", page_icon="🎬")
st.title("IMDB Movie Review Classifier by Anu")

# Load IMDB test data
(_, _), (xtest, ytest) = imdb.load_data(num_words=10000)

st.header("5 Sample Test Reviews")
for i in range(5):
    seq = xtest[i]
    text = decode_review(seq)
    prob = predict_prob(seq)
    pred = 'Positive' if prob >= 0.5 else 'Negative'
    actual = 'Positive' if ytest[i] == 1 else 'Negative'

    st.subheader(f"Sample {i+1}")
    st.write("Actual:", actual)
    st.write(f"Predicted: {pred} (prob={prob:.4f})")
    st.write("Review (truncated):", text[:600])
    st.markdown("---")


