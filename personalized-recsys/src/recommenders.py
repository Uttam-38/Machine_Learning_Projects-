import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

class PopularityRecommender:
    def __init__(self):
        self.pop_ranked: List[int] = []

    def fit(self, train_ratings: pd.DataFrame) -> None:
        # Popularity by count, tie-break by avg rating
        stats = train_ratings.groupby("movieId").agg(
            cnt=("rating", "count"),
            avg=("rating", "mean")
        ).reset_index()
        stats = stats.sort_values(["cnt", "avg"], ascending=False)
        self.pop_ranked = stats["movieId"].tolist()

    def recommend(self, user_id: int, seen: Set[int], k: int) -> List[int]:
        recs = [m for m in self.pop_ranked if m not in seen]
        return recs[:k]


class ContentBasedRecommender:
    """
    Content similarity between items using TF-IDF on text features.
    We'll build an item-item similarity matrix implicitly via cosine similarity.
    For user recommendations: score items by similarity to user's liked/watched items.
    """
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.movie_ids: Optional[np.ndarray] = None
        self.movie_id_to_idx: Dict[int, int] = {}

    def _build_item_text(self, movies_df: pd.DataFrame) -> pd.Series:
        # Combine title + genres into a single text field
        # (simple but effective)
        text = (movies_df["title"].fillna("") + " " + movies_df["genres"].fillna("")).astype(str)
        return text

    def fit(self, movies_df: pd.DataFrame) -> None:
        self.movie_ids = movies_df["movieId"].values
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}

        text = self._build_item_text(movies_df)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2)
        )
        X = self.vectorizer.fit_transform(text)
        # Normalize TF-IDF vectors so dot(X, X.T) becomes cosine similarity
        self.item_embeddings = normalize(X).astype(np.float32)

    def score_items_for_user(self, user_history: List[int]) -> np.ndarray:
        """
        Returns scores for all items (aligned with self.movie_ids)
        based on similarity to user's history items.
        """
        if self.item_embeddings is None or self.movie_ids is None:
            raise RuntimeError("ContentBasedRecommender not fitted.")

        idxs = [self.movie_id_to_idx[m] for m in user_history if m in self.movie_id_to_idx]
        if not idxs:
            # cold-start user: no history => scores = 0
            return np.zeros(len(self.movie_ids), dtype=np.float32)

        # Average similarity to user's items:
        # score(item) = mean cosine(item, history_item)
        # Implemented as dot between item vectors and mean history vector.
        hist_vec = self.item_embeddings[idxs].mean(axis=0)
        scores = (self.item_embeddings @ hist_vec.T).A1  # convert 1xN sparse to array
        return scores.astype(np.float32)

    def recommend(self, user_history: List[int], seen: Set[int], k: int) -> List[int]:
        scores = self.score_items_for_user(user_history)
        # Rank by score desc
        ranked_idxs = np.argsort(-scores)
        recs = []
        for idx in ranked_idxs:
            mid = int(self.movie_ids[idx])
            if mid not in seen:
                recs.append(mid)
            if len(recs) >= k:
                break
        return recs


class CollaborativeSVDRecommender:
    """
    Simple implicit-style collaborative filtering:
    - Build user-item matrix from explicit ratings
    - Apply TruncatedSVD
    - Predict scores via reconstructed matrix

    This is a classic, lightweight approach suitable for internships + GitHub.
    """
    def __init__(self, n_components: int = 50):
        self.n_components = n_components

        self.user_ids: Optional[np.ndarray] = None
        self.movie_ids: Optional[np.ndarray] = None
        self.user_id_to_idx: Dict[int, int] = {}
        self.movie_id_to_idx: Dict[int, int] = {}

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

    def fit(self, train_ratings: pd.DataFrame) -> None:
        users = np.sort(train_ratings["userId"].unique())
        movies = np.sort(train_ratings["movieId"].unique())

        self.user_ids = users
        self.movie_ids = movies
        self.user_id_to_idx = {u: i for i, u in enumerate(users)}
        self.movie_id_to_idx = {m: j for j, m in enumerate(movies)}

        # Build dense matrix (for ML-1M this is ok-ish; for huge scale use sparse)
        n_u, n_m = len(users), len(movies)
        R = np.zeros((n_u, n_m), dtype=np.float32)

        self.global_mean = float(train_ratings["rating"].mean())

        # Fill matrix with centered ratings
        for row in train_ratings.itertuples(index=False):
            ui = self.user_id_to_idx[int(row.userId)]
            mj = self.movie_id_to_idx[int(row.movieId)]
            R[ui, mj] = float(row.rating) - self.global_mean

        svd = TruncatedSVD(n_components=min(self.n_components, min(R.shape) - 1), random_state=42)
        U = svd.fit_transform(R)                   # (n_users, k)
        Vt = svd.components_                       # (k, n_items)

        # Factors
        self.user_factors = U.astype(np.float32)
        self.item_factors = Vt.T.astype(np.float32)  # (n_items, k)

    def score_all_items(self, user_id: int) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("CollaborativeSVDRecommender not fitted.")
        if self.user_ids is None or self.movie_ids is None:
            raise RuntimeError("Model ids not available.")

        if user_id not in self.user_id_to_idx:
            # cold-start user: return zeros
            return np.zeros(len(self.movie_ids), dtype=np.float32)

        ui = self.user_id_to_idx[user_id]
        scores = self.user_factors[ui] @ self.item_factors.T  # centered scores
        scores = scores + self.global_mean
        return scores.astype(np.float32)

    def recommend(self, user_id: int, seen: Set[int], k: int) -> List[int]:
        scores = self.score_all_items(user_id)
        ranked = np.argsort(-scores)

        recs = []
        for idx in ranked:
            mid = int(self.movie_ids[idx])
            if mid not in seen:
                recs.append(mid)
            if len(recs) >= k:
                break
        return recs


class HybridRecommender:
    """
    Blend CF + Content + Popularity fallback.
    """
    def __init__(self, cf: CollaborativeSVDRecommender, cb: ContentBasedRecommender, pop: PopularityRecommender,
                 cf_weight: float = 0.65, content_weight: float = 0.35):
        self.cf = cf
        self.cb = cb
        self.pop = pop
        self.cf_weight = cf_weight
        self.content_weight = content_weight

        self.all_movie_ids: Optional[np.ndarray] = None

    def fit(self, train_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        self.pop.fit(train_ratings)
        self.cb.fit(movies_df)
        self.cf.fit(train_ratings)

        self.all_movie_ids = movies_df["movieId"].values

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-8:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def recommend(self, user_id: int, user_history: List[int], seen: Set[int], k: int) -> List[int]:
        # If we can't score, fallback to popularity
        # CF scores only items in CF movie universe; content scores all items in movies_df
        try:
            cf_scores = self.cf.score_all_items(user_id)  # aligned to self.cf.movie_ids
        except Exception:
            cf_scores = None

        cb_scores = self.cb.score_items_for_user(user_history)  # aligned to movies_df movie ids

        # Cold-start user -> mostly content/popularity
        cold_user = (len(user_history) == 0) or (user_id not in self.cf.user_id_to_idx)

        # Build a unified score dict over movies_df universe
        # Map cf scores into the full universe
        scores = {}

        # Content
        cb_scores_norm = self._minmax(cb_scores)
        for idx, mid in enumerate(self.cb.movie_ids):
            scores[int(mid)] = self.content_weight * float(cb_scores_norm[idx])

        # CF contribution
        if (cf_scores is not None) and (not cold_user):
            cf_scores_norm = self._minmax(cf_scores)
            for idx, mid in enumerate(self.cf.movie_ids):
                scores[int(mid)] = scores.get(int(mid), 0.0) + self.cf_weight * float(cf_scores_norm[idx])

        # If cold user, boost popularity as fallback
        if cold_user:
            # Add small popularity prior
            pop_list = self.pop.pop_ranked[:5000]  # cap
            # higher rank => higher score
            for rank, mid in enumerate(pop_list, start=1):
                if mid in scores:
                    scores[mid] += 0.10 * (1.0 / rank)

        # Rank
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        recs = []
        for mid, _ in ranked:
            if mid not in seen:
                recs.append(mid)
            if len(recs) >= k:
                break

        # Final fallback: popularity
        if len(recs) < k:
            recs_extra = self.pop.recommend(user_id=user_id, seen=set(seen).union(set(recs)), k=k - len(recs))
            recs.extend(recs_extra)

        return recs
