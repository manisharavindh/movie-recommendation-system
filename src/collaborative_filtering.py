# src/collaborative_filtering.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

class UserBasedCollaborativeFilter:
    def __init__(self, n_neighbors=50, similarity_threshold=0.1):
        self.n_neighbors = n_neighbors
        self.similarity_threshold = similarity_threshold
        self.user_similarity = None
        self.user_item_matrix = None
        self.user_means = None
        
    def create_user_item_matrix(self, ratings_df):
        """Create user-item matrix from ratings dataframe"""
        self.user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        return self.user_item_matrix
    
    def calculate_user_similarity(self):
        """Calculate cosine similarity between users"""
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
    def fit(self, ratings_df):
        """Train the collaborative filtering model"""
        print("Creating user-item matrix...")
        self.create_user_item_matrix(ratings_df)
        
        print("Calculating user similarities...")
        self.calculate_user_similarity()
        
        # Calculate user means for mean-centered predictions
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        print("Model training complete!")
        
    def predict_rating(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()  # Global average
            
        if movie_id not in self.user_item_matrix.columns:
            return self.user_means[user_id]  # User average
        
        # Find similar users who rated this movie
        user_similarities = self.user_similarity[user_id]
        movie_ratings = self.user_item_matrix[movie_id]
        
        # Get users who rated this movie
        rated_users = movie_ratings[movie_ratings > 0].index
        similar_users = user_similarities[rated_users]
        
        # Filter by similarity threshold
        similar_users = similar_users[similar_users > self.similarity_threshold]
        
        if len(similar_users) == 0:
            return self.user_means[user_id]
        
        # Calculate weighted average
        numerator = sum(similar_users * movie_ratings[similar_users.index])
        denominator = sum(similar_users)
        
        return numerator / denominator if denominator > 0 else self.user_means[user_id]
    
    def recommend_movies(self, user_id, n_recommendations=10):
        """Get top N movie recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get movies user hasn't rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            pred_rating = self.predict_rating(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]