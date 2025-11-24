import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import tensorflow as tf

# Load LSTM SavedModel
lstm_model = tf.keras.models.load_model("lstm_imdb_savedmodel", compile=False)
serve_fn = lstm_model.signatures['serving_default']

# Load IMDB word index and decoder
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

# Streamlit UI
st.set_page_config(page_title="IMDB Movie Review Classifier", page_icon="🎬")
st.title("IMDB Movie Review Classifier by Anu")

# Load IMDB test data
(_, _), (xtest, ytest) = imdb.load_data(num_words=10000)

st.header("5 Sample Test Reviews")
for i in range(5):
    seq = xtest[i]
    text = decode_review(seq)

    # Pad and convert to tensor
    seq_padded = pad_sequences([seq], maxlen=200, padding='post')
    seq_tensor = tf.constant(seq_padded, dtype=tf.float32)

    # Call the SavedModel serving function
    # NOTE: check the input name from your model signature
    input_name = list(serve_fn.structured_input_signature[1].keys())[0]
    prob_tensor = serve_fn(**{input_name: seq_tensor})

    # Extract probability from output tensor
    output_name = list(prob_tensor.keys())[0]
    prob = prob_tensor[output_name].numpy()[0,0]

    pred = 'Positive' if prob >= 0.5 else 'Negative'
    actual = 'Positive' if ytest[i] == 1 else 'Negative'

    st.subheader(f"Sample {i+1}")
    st.write("Actual:", actual)
    st.write(f"Predicted: {pred} (prob={prob:.4f})")
    st.write("Review (truncated):", text[:600])
    st.markdown("---")
