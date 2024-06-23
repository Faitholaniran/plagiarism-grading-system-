import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Dense, Bidirectional, LSTM
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load the spacy model for synonym detection
nlp = spacy.load('en_core_web_md')

# Define the model architecture
def create_model(vocab_size, maxlen):
    embedding_dims = 100
    filters = 128
    kernel_size = 5
    hidden_dims = 128

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dims, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Bidirectional(LSTM(hidden_dims, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the pre-trained model
def load_plagiarism_model(model_path):
    model = load_model(model_path)
    return model

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to get embeddings from multiple models
def get_embeddings(models, texts):
    embeddings = []
    for model in models:
        embeddings.append(model.encode(texts))
    return embeddings

# Function to calculate exact word match score
def exact_word_match_score(text1, text2):
    words1 = text1.split()
    words2 = text2.split()
    counter1 = Counter(words1)
    counter2 = Counter(words2)
    common_words = counter1 & counter2
    common_word_count = sum(common_words.values())
    total_words = len(words1) + len(words2)
    return (2 * common_word_count) / total_words

# Function to calculate synonym match score
def synonym_match_score(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    synonyms1 = set(token.lemma_ for token in doc1)
    synonyms2 = set(token.lemma_ for token in doc2)
    common_synonyms = synonyms1 & synonyms2
    total_synonyms = len(synonyms1) + len(synonyms2)
    return (2 * len(common_synonyms)) / total_synonyms

# Function to check for plagiarism
def check_plagiarism(model_path, assignment_text, target_texts, plagiarism_threshold=70.0):
    model = load_plagiarism_model(model_path)
    
    # Pre-trained sentence transformer models
    transformer_models = [
        SentenceTransformer('paraphrase-mpnet-base-v2'),
        SentenceTransformer('paraphrase-MiniLM-L6-v2'),
        SentenceTransformer('distiluse-base-multilingual-cased-v2')
    ]
    
    assignment_text = preprocess_text(assignment_text)
    target_texts = [preprocess_text(text) for text in target_texts]

    # Encode assignment text and target essays into vectors using multiple models
    assignment_embeddings = get_embeddings(transformer_models, [assignment_text])
    target_embeddings = get_embeddings(transformer_models, target_texts)

    # Calculate cosine similarities for each model
    similarities = []
    for i in range(len(transformer_models)):
        assignment_vector = assignment_embeddings[i][0]
        target_vectors = target_embeddings[i]
        similarity = cosine_similarity([assignment_vector], target_vectors)[0]
        similarities.append(similarity)

    # Average the similarities from different models
    average_similarities = np.mean(similarities, axis=0)

    # Calculate exact word match scores
    word_match_scores = [exact_word_match_score(assignment_text, target_text) for target_text in target_texts]

    # Calculate synonym match scores
    synonym_match_scores = [synonym_match_score(assignment_text, target_text) for target_text in target_texts]

    # Combine semantic similarity, exact word match score, and synonym match score
    final_scores = 0.4 * average_similarities + 0.3 * np.array(word_match_scores) + 0.3 * np.array(synonym_match_scores)

    # Print similarity percentages
    for i, score in enumerate(final_scores):
        percentage = score * 100
        print(f"Similarity with target text {i+1}: {percentage:.2f}%")

    # Check for plagiarism
    plagiarized = any(score * 100 >= plagiarism_threshold for score in final_scores)
    
    if plagiarized:
        print("\nThe assignment contains plagiarism.")
        return True
    else:
        print("\nNo plagiarism detected.")
        return False

# Example usage:
if __name__ == "__main__":
    # Define the path to your model
    model_path = './essay_model_lstm_cnn.h5'

    # Example assignment text and target texts
    assignment_text = "This section details the implementation of the proposed intelligent auto-response system. It presents the hardware and software requirements required to make use of the system, the development methodology, program modules and interfaces and the evaluation of the system."
    
    # You would typically fetch these target texts from your database
    target_texts = [
        """
        This section details the implementation of the proposed intelligent auto-response system. It presents the hardware and software requirements required to make use of the system, the development methodology, program modules and interfaces and the evaluation of the system.
        """,
        """
        I hereby certify that this project, was carried out by Olubusolami Rosemary SOGUNLE in the Department of Computer and Information Sciences, College of Science and Technology, Covenant University, Ogun State, Nigeria, under my supervision.
        """,
        """
        The proposed system will be a web-based email client running on the Microsoft Outlook server. The application will allow users to receive new emails and send replies. The replies sent will either be generated by them or the email client. For intelligently generated email responses, they can choose to allow
        """
    ]

    # Check for plagiarism
    check_plagiarism(model_path, assignment_text, target_texts)
