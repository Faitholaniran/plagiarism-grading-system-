
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os
import nltk
import language_tool_python
from nltk.tokenize import word_tokenize
from textstat import textstat

tool = language_tool_python.LanguageTool('en-US')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
W2V_SIZE = 300
  # Load the Word2Vec model
from gensim.models import Word2Vec
from nltk.corpus import stopwords, wordnet as wn
w2v_model = Word2Vec.load('word2vec.model')

# Define stop words
stop_words = set(stopwords.words("english"))
stop_words.remove('not')
more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
stop_words = stop_words.union(more_stopwords)
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

def detailed_spelling_grammar_check(essay):
    matches = tool.check(essay)
    grammar_errors = 0
    spelling_errors = 0
    for match in matches:
        if 'Spelling' in match.ruleIssueType:
            spelling_errors += 1
        else:
            grammar_errors += 1
    return spelling_errors, grammar_errors


# Function to make predictions
def predict(text, model, w2v_model, stop_words):
    # Preprocess the text
    input_vec = preprocess_text(text, w2v_model, stop_words)
    # Predict using the model
    prediction = model.predict(input_vec)
    return np.round(prediction).flatten()[0]

def predict_word(essay, total_score=12, topics=None, topic_description=None):
    # Example usage:
  
    
    # Sample text for prediction

    # Get the prediction
    predicted_score = predict(essay, model, w2v_model, stop_words)
    spelling_errors, grammar_errors = detailed_spelling_grammar_check(essay)
    readability_score_value = textstat.flesch_reading_ease(essay)
    if readability_score_value > 60:
        readability_level = 'Easy'
    elif readability_score_value > 30:
        readability_level = 'Medium'
    else:
        readability_level = 'Difficult'
    # Check if the essay covers the specified topics
    topics_covered = any(topic.lower() in essay.lower() for topic in topics)

    # Check if the essay is in line with the topic description
    description_match = True
    if topic_description:
        description_words = set(topic_description.lower().split())
        essay_words = set(essay.lower().split())
        description_match = len(description_words.intersection(essay_words)) > 0

    # Calculate additional scoring factors
    word_count_factor = min(len(word_tokenize(essay)) / 100, 1.0) * 5
    coherence_score = 5 if topics_covered else 0
    topic_alignment_score = 5 if description_match else -5


    # Calculate the final score
    final_score = max(0, readability_score_value - grammar_errors - spelling_errors + coherence_score + word_count_factor + topic_alignment_score)
    normalized_score = min(final_score, total_score)  # Ensure normalized score is <= total_score

    # Create feedback dictionary
    feedback = {
        "predicted_score": normalized_score,
        "total_score": total_score,
        "spelling_errors": spelling_errors,
        "grammar_errors": grammar_errors,
        "readability_score": readability_score_value,
        "readability_level": readability_level,
        "topics_covered": topics_covered,
        "description_match": description_match
    }

    return feedback

print(predict_word("essay things that need to essay are essaying today oooo", 100, ["science"], "scientific"))