
import os
import pandas as pd
import numpy as np
import nltk
import re
import spacy
import language_tool_python
from textstat import textstat
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import language_tool_python
from textstat import textstat
import spacy
from keras.models import load_model
lstm_model = load_model('final_lstm_model.h5')

# Load SpaCy model and LanguageTool
nlp = spacy.load('en_core_web_sm')
tool = language_tool_python.LanguageTool('en-US')
W2V_SIZE = 300

def essay_to_wordlist(essay, remove_stopwords=False):
    words = word_tokenize(essay)
    if remove_stopwords:
        words = [word for word in words if word not in stop_words]
    return words

def correct_spelling_grammar(text):
    matches = tool.check(text)
    corrected_text = tool.correct(text)
    return corrected_text, matches

def readability_score(text):
    return textstat.flesch_reading_ease(text)

def introduce_spelling_mistakes(text):
    words = text.split()
    for i in range(len(words)):
        if np.random.rand() < 0.1:
            words[i] = words[i][::-1]
    return ' '.join(words)

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def check_topic(essay, topics):
    expanded_topics = set(topics)
    for topic in topics:
        expanded_topics.update(get_synonyms(topic))
    doc = nlp(essay.lower())
    matches = []
    for token in doc:
        for topic in expanded_topics:
            if token.lemma_ == topic:
                matches.append(token.text)
                break
    return len(matches) > 0

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

def getAvgFeatureVecs(essays, model, num_features):
    # Calculate the average feature vector for each essay
    counter = 0
    featureVecs = np.zeros((len(essays), num_features), dtype="float32")
    for essay in essays:
        featureVecs[counter] = np.mean([model.wv[word] for word in essay if word in model.wv], axis=0)
        counter += 1
    return featureVecs

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

def predict_essay_score(essay, total_score=12, topics=None, topic_description=None):
    # Preprocess the essay
    clean_essay = essay_to_wordlist(essay, remove_stopwords=True)
    essay_vector = getAvgFeatureVecs([clean_essay], w2v_model, W2V_SIZE)
    essay_vector = np.reshape(essay_vector, (1, 1, W2V_SIZE))

    # Predict the essay score using the LSTM model
    predicted_score = lstm_model.predict(essay_vector)
    predicted_score = np.around(predicted_score).flatten()[0]

    # Check spelling and grammar errors
    spelling_errors, grammar_errors = detailed_spelling_grammar_check(essay)

    # Calculate readability score and level
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
