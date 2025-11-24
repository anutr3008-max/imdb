import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

# Load the SavedModel (TensorFlow format)
lstm_model = load_model("lstm_imdb_savedmodel", compile=False)

# Access the 'serve' function signature
serve_fn = lstm_model.signatures["serving_default"]  # or "serve" if exported differently

# Load word index
word_index = imdb.get_word_index()
reverse_word_index = {value+3: key for (key, value) in word_index.items()}
reverse_word_index[0] = '<PAD>'
reverse_word_index[1] = '<START>'
reverse_word_index[2] = '<UNK>'
reverse_word_index[3] = 'the'

def decode_review(seq):
    return ' '.join([reverse_word_index.get(i, '?') for i in seq if i != 0])

# Streamlit UI
st.set_page_config(page_title="IMDB Movie Review Classifier", page_icon="🎬")
st.title("IMDB Movie Review Classifier by Anu")

(_, _), (xtest, ytest) = imdb.load_data(num_words=10000)

st.header("5 Sample Test Reviews")
for i in range(5):
    seq = xtest[i]
    seq_padded = pad_sequences([seq], maxlen=500, padding='post')  # must match SavedModel input length
    seq_tensor = tf.constant(seq_padded, dtype=tf.float32)

    # Call the SavedModel using the signature
    prob_tensor = serve_fn(args_0=seq_tensor)  
    prob = prob_tensor[list(prob_tensor.keys())[0]].numpy()[0,0]

    pred = 'Positive' if prob >= 0.5 else 'Negative'
    actual = 'Positive' if ytest[i] == 1 else 'Negative'

    st.subheader(f"Sample {i+1}")
    st.write("Actual:", actual)
    st.write(f"Predicted: {pred} (prob={prob:.4f})")
    st.write("Review (truncated):", decode_review(seq)[:600])
    st.markdown("---")



# Optional: User input for custom review
st.header("Try Your Own Review")
user_input = st.text_area("Enter a movie review here:")

if st.button("Predict"):
    if user_input.strip() != "":
        encoded_input = encode_review(user_input)
        prob = lstm_model(seq_padded, training=False).numpy()[0,0]
        pred = 'Positive' if prob >= 0.5 else 'Negative'
        st.write(f"Prediction: {pred} (prob={prob:.4f})")
    else:
        st.write("Please enter a review first.")







