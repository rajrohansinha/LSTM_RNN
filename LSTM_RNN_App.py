import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.set_page_config(page_title="Next Word Prediction", layout="centered")

# Sidebar
st.sidebar.title("About")
st.sidebar.markdown("""
This app uses an LSTM model to predict the next word in a given sequence of text.  
- **Model**: Pre-trained LSTM  
- **Tokenizer**: Pre-saved tokenizer for text preprocessing  
""")
st.sidebar.info("Enter a sequence of words in the input box and click 'Predict' to see the next word.")

# Main UI
st.title("Next Word Prediction With LSTM")
st.markdown("### Enter a sequence of words below and let the model predict the next word!")

# Input text
input_text = st.text_input("Enter the sequence of Words", placeholder="Type a sentence like 'To be or not to'")

# Predict button
if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.error("Please enter a valid sequence of words.")
    else:
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.success(f"**Next word:** {next_word}")
        else:
            st.warning("Could not predict the next word. Try a different input.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and TensorFlow.")