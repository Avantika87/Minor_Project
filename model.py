import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
nltk.download('stopwords')
nltk.download('punkt')
import joblib

data = pd.read_csv('./aitools.csv')
data.head(5)

data.shape

# Create a new column to store the original descriptions
data['pre_Description'] = data['Description']

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Preprocess text data
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is not NaN
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = word_tokenize(text)  # Tokenize
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return ' '.join(words)
    else:
        return ''  # Return empty string for NaN



data['New_Description'] = data['pre_Description'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['New_Description'])

# Train Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
knn_model.fit(tfidf_matrix)

#Saving our model for future
joblib.dump(knn_model,"Recommender.pkl")

# User input description
user_description = "To classify things in diffent catogory and predict the outcome"

user_input = preprocess_text(user_description)
user_input_vector = tfidf_vectorizer.transform([user_input])

# Find nearest neighbors
distances, indices = knn_model.kneighbors(user_input_vector)

# recommended_tools = pd.DataFrame(columns=["Tool Name", "Description"])
recommended_tools = []

# Extract the desired columns from the original data
selected_columns = ['AI Tool Name', 'Free/Paid/Other', 'Charges', 'Languages', 'Major Category','Tool Link','Description']

# Create a DataFrame to store the selected data
recommended_tools_with_info = pd.DataFrame(columns=selected_columns)

# Iterate through the recommended indices and add the selected data to the DataFrame
for dist, i in zip(distances[0], indices[0]):
    tool_data = data.loc[i, selected_columns]
    recommended_tools_with_info = recommended_tools_with_info.append(tool_data, ignore_index=True)

print(recommended_tools_with_info)

