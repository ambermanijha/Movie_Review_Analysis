import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the sentiment analysis model
model = load_model('sentiment_imbd.h5')

# Set parameters for tokenization and padding (these should match how the model was trained)
max_length = 100  # Maximum length of the review (depends on model training)
vocab_size = 10000  # Vocabulary size used in training (adjust based on your tokenizer)

# Create a tokenizer (this should match the tokenizer used during model training)
tokenizer = Tokenizer(num_words=vocab_size)

# Create a Streamlit app
st.title("Movie Review Sentiment Analysis")

# Get the movie name and review from the user
movie_name = st.text_input("Enter the movie name:")
review = st.text_area("Enter your review:")

# Create a button to submit the review
if st.button("Submit"):
    # Preprocess the review text: tokenize and pad the sequence
    review_text = [review]
    sequences = tokenizer.texts_to_sequences(review_text)
    padded_review = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Make predictions using the model
    predictions = model.predict(padded_review)
    
    # Get the sentiment score
    sentiment_score = np.argmax(predictions[0])
    
    # Determine if the movie is recommended
    if sentiment_score == 0:
        st.write("The movie is not recommended.")
    else:
        st.write("The movie is recommended.")
