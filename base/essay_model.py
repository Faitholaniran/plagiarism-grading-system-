
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os
import nltk
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')
nltk.download('stopwords')
# Function to recreate the LSTM-based model architecture
def create_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=(1, 300), return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    return model

# Create the model architecture
model = create_model()
if not os.path.exists('./final_lstm_model.h5'):
    print("Hi there")
# Load weights from the .h5 file
weights_path = './final_lstm_model.h5'
model.load_weights(weights_path)
# Helper functions
def essay_to_wordlist(essay, remove_stopwords=False):
    words = word_tokenize(essay)
    if remove_stopwords:
        words = [word for word in words if word not in stop_words]
    return words

def correct_spelling_grammar(text):
    matches = tool.check(text)
    corrected_text = tool.correct(text)
    return corrected_text, matches

# Function to preprocess input text into feature vectors
def preprocess_text(text, w2v_model, stop_words):
    # Correct spelling and grammar
    corrected_text, _ = correct_spelling_grammar(text)
    # Tokenize and clean text
    words = essay_to_wordlist(corrected_text, remove_stopwords=True)
    # Convert words to vectors
    word_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    # Average the vectors
    if word_vecs:
        feature_vec = np.mean(word_vecs, axis=0)
    else:
        feature_vec = np.zeros((W2V_SIZE,), dtype="float32")
    # Reshape for model input
    return feature_vec.reshape(1, 1, W2V_SIZE)

# Function to make predictions
def predict(text, model, w2v_model, stop_words):
    # Preprocess the text
    input_vec = preprocess_text(text, w2v_model, stop_words)
    # Predict using the model
    prediction = model.predict(input_vec)
    return np.round(prediction).flatten()[0]

# Example usage:
# Load the Word2Vec model
from gensim.models import Word2Vec
w2v_model = Word2Vec.load('word2vec.model')

# Define stop words
stop_words = set(nltk.corpus.stopwords.words("english"))
stop_words.remove('not')
more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
stop_words = stop_words.union(more_stopwords)

# Sample text for prediction
sample_text = "This is a sample essay text that needs to be graded."

# Get the prediction
predicted_score = predict(sample_text, model, w2v_model, stop_words)
print(f"Predicted Score: {predicted_score}")
