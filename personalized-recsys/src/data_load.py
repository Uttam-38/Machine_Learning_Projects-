import os
import pandas as pd
from typing import Tuple

def load_movielens(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      ratings_df columns: [userId, movieId, rating, timestamp]
      movies_df columns:  [movieId, title, genres]
    Supports MovieLens 1M (ratings.dat, movies.dat) and 100K (u.data, u.item).
    """
    # MovieLens 1M format
    ratings_1m = os.path.join(data_dir, "ratings.dat")
    movies_1m = os.path.join(data_dir, "movies.dat")

    # MovieLens 100K format
    ratings_100k = os.path.join(data_dir, "u.data")
    movies_100k = os.path.join(data_dir, "u.item")

    if os.path.exists(ratings_1m) and os.path.exists(movies_1m):
        ratings = pd.read_csv(
            ratings_1m,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="latin-1",
        )
        movies = pd.read_csv(
            movies_1m,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        return ratings, movies

    if os.path.exists(ratings_100k) and os.path.exists(movies_100k):
        ratings = pd.read_csv(
            ratings_100k,
            sep="\t",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="latin-1",
        )
        movies = pd.read_csv(
            movies_100k,
            sep="|",
            names=[
                "movieId", "title", "release_date", "video_release_date", "imdb_url",
                "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
                "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
                "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"
            ],
            encoding="latin-1",
        )
        genre_cols = [
            "unknown","Action","Adventure","Animation","Childrens","Comedy","Crime","Documentary",
            "Drama","Fantasy","FilmNoir","Horror","Musical","Mystery","Romance","SciFi","Thriller","War","Western"
        ]
        # Convert one-hot genres into "Action|Comedy|..."
        def row_to_genres(row):
            gs = [g for g in genre_cols if row.get(g, 0) == 1]
            return "|".join(gs) if gs else "unknown"

        movies["genres"] = movies.apply(row_to_genres, axis=1)
        movies = movies[["movieId", "title", "genres"]]
        return ratings, movies

    raise FileNotFoundError(
        f"Could not find MovieLens files in {data_dir}. "
        "Place ratings.dat/movies.dat (1M) or u.data/u.item (100K) into data/raw/."
    )
