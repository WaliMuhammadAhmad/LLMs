import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prettytable import PrettyTable

# Load your dataset
df = pd.read_csv('/kaggle/input/llms-details/llm.csv')

# Preprocessing
df['name'] = df['name'].str.lower()
df['owner'] = df['owner'].str.lower()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
name_vectors = vectorizer.fit_transform(df['name'])

def get_model_details(user_input):
    # Preprocess user input
    user_input = user_input.lower()
    
    # Vectorize user input
    user_vector = vectorizer.transform([user_input])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, name_vectors).flatten()
    
    # Find the index of the most similar model
    most_similar_index = similarities.argmax()
    
    # Get the details of the most similar model
    model_details = df.iloc[most_similar_index]
    
    return model_details

# Example usage:
user_input = input("Enter Model Name to see the details: ")
result = get_model_details(user_input)

# Print the result in a table
table = PrettyTable()
table.field_names = ["Model LLM", "Owner", "Train on x billion parameters", "Date of Release"]
table.add_row([result['name'], result['owner'], result['trained on x billion parameters'], result['date']])
print("\nModel Details:")
print(table)