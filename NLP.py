import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# Sample intents (can be expanded based on use case)
intents = {
    "greetings": {
        "patterns": ["Hello", "Hi", "Hey", "Good day", "Greetings", "What's up?"],
        "responses": ["Hello! How can I help you today?", "Hi there!", "Hey! How are you doing?"]
    },
    "goodbye": {
        "patterns": ["Goodbye", "See you later", "Bye", "Take care"],
        "responses": ["Goodbye!", "See you later!", "Have a great day!"]
    },
    "thanks": {
        "patterns": ["Thanks", "Thank you", "That's helpful"],
        "responses": ["You're welcome!", "Happy to help!", "Anytime!"]
    },
    "no_answer": {
        "patterns": [],
        "responses": ["I'm sorry, I don't understand.", "Can you please rephrase?"]
    }
}

# Preprocess input sentence
def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return tokens

# Create a response based on similarity of the user's input to known patterns
def chatbot_response(user_input):
    # Preprocess the input
    user_input_tokens = preprocess(user_input)

    # Create a list of all patterns across all intents
    all_patterns = []
    intent_keys = []
    for intent, data in intents.items():
        for pattern in data['patterns']:
            all_patterns.append(' '.join(preprocess(pattern)))
            intent_keys.append(intent)

    # Vectorize the user input and the patterns
    vectorizer = CountVectorizer().fit_transform([user_input] + all_patterns)
    vectors = vectorizer.toarray()

    # Compute similarity between user input and all patterns
    cosine_sim = cosine_similarity(vectors[0].reshape(1, -1), vectors[1:])

    # Find the best matching pattern
    best_match_index = np.argmax(cosine_sim)
    best_match_intent = intent_keys[best_match_index]

    # If similarity is above a threshold, return a random response from the matched intent
    if cosine_sim[0, best_match_index] > 0.1:
        return random.choice(intents[best_match_intent]['responses'])
    else:
        return random.choice(intents['no_answer']['responses'])

# Chatbot loop
print("AI Chatbot: Hi! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("AI Chatbot: Goodbye!")
        break
    else:
        response = chatbot_response(user_input)
        print("AI Chatbot:", response)
