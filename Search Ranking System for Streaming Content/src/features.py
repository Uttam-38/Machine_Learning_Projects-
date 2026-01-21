import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.config import ensure_dirs
from src.utils import safe_div

def parse_genres(s: str):
    return s.split("|") if isinstance(s, str) else []

def build_user_genre_affinity(df_sessions: pd.DataFrame):
    tmp = df_sessions.copy()
    tmp["genre_list"] = tmp["genres"].apply(parse_genres)
    expl = tmp.explode("genre_list")
    # clicked+dwell acts as implicit feedback weight
    expl["implicit_weight"] = 0.4 * expl["clicked"] + 0.6 * np.clip(expl["dwell_time"] / 120.0, 0, 1)
    g = expl.groupby(["user_id", "genre_list"]).agg(
        w=("implicit_weight", "mean"),
        c=("clicked", "sum")
    ).reset_index()
    # normalize per user
    g["genre_affinity"] = g.groupby("user_id")["w"].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
    return g[["user_id", "genre_list", "genre_affinity"]]

def main():
    paths = ensure_dirs()
    sess_path = paths.processed / "sessions.parquet"
    df = pd.read_parquet(sess_path)

    # TF-IDF relevance: query vs title
    titles = df["title"].fillna("").astype(str).tolist()
    queries = df["query"].fillna("").astype(str).tolist()

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    tfidf_titles = tfidf.fit_transform(titles)
    tfidf_queries = tfidf.transform(queries)

    # cosine similarity for sparse vectors: dot product if vectors are L2-normalized (TFIDF is normalized by default in sklearn)
    text_sim = (tfidf_titles.multiply(tfidf_queries)).sum(axis=1)
    df["tfidf_cosine"] = np.asarray(text_sim).reshape(-1)

    # genre overlap feature
    df["genre_list"] = df["genres"].apply(parse_genres)
    df["query_is_genre"] = (df["query_type"] == "genre").astype(int)
    df["genre_overlap"] = df.apply(
        lambda r: int(r["query"].capitalize() in r["genre_list"]) if r["query_is_genre"] == 1 else 0,
        axis=1
    )

    # user-genre affinity feature
    uga = build_user_genre_affinity(df)
    df_expl = df.explode("genre_list").rename(columns={"genre_list": "genre"})
    uga2 = uga.rename(columns={"genre_list": "genre"})
    df_expl = df_expl.merge(uga2, on=["user_id", "genre"], how="left")
    df_expl["genre_affinity"] = df_expl["genre_affinity"].fillna(0.0)
    # aggregate affinity over item genres
    aff = df_expl.groupby(["session_id", "movie_id"])["genre_affinity"].mean().reset_index()
    df = df.merge(aff, on=["session_id", "movie_id"], how="left")
    df["genre_affinity"] = df["genre_affinity"].fillna(0.0)

    # popularity & recency proxies
    current_year = 2003  # MovieLens 1M era; keep consistent
    df["item_age"] = (current_year - df["year"]).clip(lower=0)
    df["pop_x_recency"] = df["popularity"] * (1.0 / (1.0 + df["item_age"]))

    # behavioral features (keep, but note leakage risk in real systems; here it’s simulated)
    df["click_rate_proxy"] = df.groupby("movie_id")["clicked"].transform("mean")
    df["dwell_proxy"] = df.groupby("movie_id")["dwell_time"].transform("mean")
    df["avg_rating_proxy"] = df.groupby("movie_id")["observed_rating"].transform("mean")

    # position shown (presentation bias indicator)
    df["inv_position"] = safe_div(1.0, df["position_shown"].astype(float))

    # final feature set
    feature_cols = [
        "tfidf_cosine",
        "genre_overlap",
        "genre_affinity",
        "popularity",
        "item_age",
        "pop_x_recency",
        "click_rate_proxy",
        "dwell_proxy",
        "avg_rating_proxy",
        "inv_position",
        "query_is_genre",
    ]

    # scale numeric features for stability (tree model doesn’t require it, but helps diagnostics)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    out = df[["session_id", "user_id", "query", "movie_id", "relevance"] + feature_cols].copy()
    out_path = paths.processed / "features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[OK] Wrote: {out_path} rows={len(out):,} sessions={out['session_id'].nunique():,}")

if __name__ == "__main__":
    main()
