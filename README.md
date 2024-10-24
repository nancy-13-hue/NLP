AI chatbot using Natural Language Processing (NLP) with Python and the NLTK library
How the Code Works:
Intents: It contains a dictionary with predefined intents (e.g., "greetings," "goodbye," "thanks") and their associated patterns (sample sentences) and responses.
Preprocessing: Tokenizes and lemmatizes user input to normalize the text.
Vectorization & Similarity: The chatbot computes the cosine similarity between the user input and predefined patterns using a CountVectorizer.
Response Generation: Based on the highest similarity score, the chatbot responds with an appropriate message. If the similarity is too low, it defaults to a “no_answer” response.
