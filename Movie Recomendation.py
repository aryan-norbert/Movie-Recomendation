import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense # type: ignore
import requests
import zipfile
import io

# Check TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Download the MovieLens dataset
url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

# Load the datasets
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Merge the datasets
data = pd.merge(ratings, movies, on='movieId')

# Plot the distribution of ratings
sns.histplot(data['rating'], bins=10)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Movie Ratings')
plt.show()

# Check for missing values
print(data.isnull().sum())

# Collaborative Filtering using SVD
user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
U, sigma, Vt = svds(user_item_matrix, k=50)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Content-Based Filtering using Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

print(get_recommendations('Toy Story (1995)'))

# Evaluation Metrics
def get_rmse(predictions, ground_truth):
    predictions = predictions[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return np.sqrt(mean_squared_error(predictions, ground_truth))

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_matrix = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
test_matrix = test_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
U_train, sigma_train, Vt_train = svds(train_matrix, k=50)
sigma_train = np.diag(sigma_train)
predicted_train_ratings = np.dot(np.dot(U_train, sigma_train), Vt_train)
predicted_train_ratings_df = pd.DataFrame(predicted_train_ratings, columns=train_matrix.columns, index=train_matrix.index)
train_rmse = get_rmse(predicted_train_ratings_df.values, train_matrix.values)
test_rmse = get_rmse(predicted_train_ratings_df.values, test_matrix.values)
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Neural Network-based Recommendation Model using TensorFlow
class RecommenderNet(Model):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        self.user_embedding = Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.movie_embedding = Embedding(num_movies, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.user_flatten = Flatten()
        self.movie_flatten = Flatten()
        self.dot = Dot(axes=1)

    def call(self, inputs):
        user_vector = self.user_flatten(self.user_embedding(inputs[0]))
        movie_vector = self.movie_flatten(self.movie_embedding(inputs[1]))
        return self.dot([user_vector, movie_vector])

user_ids = data['userId'].unique().astype(np.int32)
movie_ids = data['movieId'].unique().astype(np.int32)
num_users = len(user_ids)
num_movies = len(movie_ids)
embedding_size = 50
model = RecommenderNet(num_users, num_movies, embedding_size)
model.compile(optimizer='adam', loss='mse')
history = model.fit([data['userId'], data['movieId']], data['rating'], epochs=5, batch_size=64, validation_split=0.2)

