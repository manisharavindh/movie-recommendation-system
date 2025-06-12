import numpy as np
import pandas as pd

def load_data(path='../data/raw/ml-latest-small/'):
    ratings = pd.read_csv(path + 'ratings.csv')
    movies = pd.read_csv(path + 'movies.csv')
    return ratings, movies

def get_data_summary(ratings_df, movies_df):
    summary = {
        'num_users' : ratings_df['userId'].nunique(),
        'num_movies' : ratings_df['movieId'].nunique(),
        'num_ratings' : ratings_df.shape[0],
        'avg_rating' : ratings_df['rating'].mean()
    }
    return summary

def filter_sparse_data(ratings_df, min_user_ratings=20, min_movie_ratings=50):

    user_count = ratings_df['userId'].value_counts()
    movie_count = ratings_df['movieId'].value_counts()

    sparse_users = user_count[user_count >= min_user_ratings].index
    sparse_movies = movie_count[movie_count >= min_movie_ratings].index

    filtered_ratings = ratings_df[
        (ratings_df['userId'].isin(sparse_users)) & 
        (ratings_df['movieId'].isin(sparse_movies))
    ]

    return filtered_ratings