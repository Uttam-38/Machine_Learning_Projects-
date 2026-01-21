import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import ensure_dirs
from src.utils import safe_div

def parse_genres(s: str):
    return s.split("|") if isinstance(s, str) else []

def load_items_from_sessions(processed_dir):
    # sessions has title/genres/year/popularity; use it as item source
    sess = pd.read_parquet(processed_dir / "sessions.parquet")
    items = sess[["movie_id", "title", "genres", "year", "popularity"]].drop_duplicates("movie_id").copy()
    return items, sess

def build_query_candidates(query: str, items: pd.DataFrame, top_n=50):
    q = query.lower().strip()
    items = items.copy()
    items["title_l"] = items["title"].str.lower()
    # naive retrieval: substring match OR genre match
    mask = items["title_l"].str.contains(q, na=False)
    gmask = items["genres"].str.lower().str.contains(q, na=False)
    cand = items[mask | gmask].copy()
    if len(cand) == 0:
        cand = items.sort_values("popularity", ascending=False).head(top_n).copy()
    return cand.head(top_n)

def main():
    paths = ensure_dirs()

    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", type=int, default=1)
    ap.add_argument("--query", type=str, required=True)
    args = ap.parse_args()

    bundle = joblib.load(paths.models / "ranker.joblib")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    items, sess = load_items_from_sessions(paths.processed)
    cand = build_query_candidates(args.query, items, top_n=50).reset_index(drop=True)

    # Build lightweight features consistent with training
    # 1) tfidf cosine (fit vectorizer on titles from training corpus)
    titles_all = sess["title"].fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    tfidf.fit(titles_all)

    tf_title = tfidf.transform(cand["title"].fillna("").astype(str).tolist())
    tf_query = tfidf.transform([args.query] * len(cand))
    cos = (tf_title.multiply(tf_query)).sum(axis=1)
    tfidf_cosine = np.asarray(cos).reshape(-1)

    # genre overlap, query_is_genre proxy
    cand["genre_list"] = cand["genres"].apply(parse_genres)
    query_is_genre = int(any(args.query.capitalize() == g for gs in cand["genre_list"] for g in gs))
    genre_overlap = np.array([int(args.query.capitalize() in gs) for gs in cand["genre_list"]], dtype=float)

    # user personalization proxies (fallback to 0 at inference in this demo)
    genre_affinity = np.zeros(len(cand), dtype=float)

    # popularity/age proxies
    current_year = 2003
    item_age = (current_year - cand["year"].astype(int)).clip(lower=0).to_numpy(dtype=float)
    popularity = cand["popularity"].to_numpy(dtype=float)
    pop_x_recency = popularity * (1.0 / (1.0 + item_age))

    # global proxies from session logs
    tmp = sess.groupby("movie_id").agg(
        click_rate_proxy=("clicked", "mean"),
        dwell_proxy=("dwell_time", "mean"),
        avg_rating_proxy=("observed_rating", "mean"),
    ).reset_index()
    cand = cand.merge(tmp, on="movie_id", how="left").fillna(0.0)

    # position-related features not known at retrieval time; set neutral
    inv_position = np.full(len(cand), safe_div(1.0, 10.0))

    df_feat = pd.DataFrame({
        "tfidf_cosine": tfidf_cosine,
        "genre_overlap": genre_overlap,
        "genre_affinity": genre_affinity,
        "popularity": popularity,
        "item_age": item_age,
        "pop_x_recency": pop_x_recency,
        "click_rate_proxy": cand["click_rate_proxy"].to_numpy(dtype=float),
        "dwell_proxy": cand["dwell_proxy"].to_numpy(dtype=float),
        "avg_rating_proxy": cand["avg_rating_proxy"].to_numpy(dtype=float),
        "inv_position": inv_position,
        "query_is_genre": np.full(len(cand), float(query_is_genre)),
    })

    # Ensure same column order
    df_feat = df_feat[feature_cols]
    scores = model.predict(df_feat.to_numpy())

    cand_out = cand[["movie_id", "title", "genres", "year"]].copy()
    cand_out["score"] = scores
    cand_out = cand_out.sort_values("score", ascending=False).head(10)

    print("\nTop results:")
    for i, r in enumerate(cand_out.itertuples(index=False), start=1):
        print(f"{i:02d}. {r.title} ({r.year})  | score={r.score:.4f} | genres={r.genres}")

if __name__ == "__main__":
    main()
