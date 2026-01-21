import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ensure_dirs, SessionConfig
from src.utils import set_seed, sigmoid

TITLE_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

def load_movielens_1m(raw_dir):
    # MovieLens 1M uses :: delimiter
    movies = pd.read_csv(
        raw_dir / "ml-1m" / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )
    ratings = pd.read_csv(
        raw_dir / "ml-1m" / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )
    users = pd.read_csv(
        raw_dir / "ml-1m" / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1"
    )
    return users, movies, ratings

def extract_year(title: str):
    m = re.search(r"\((\d{4})\)", title)
    return int(m.group(1)) if m else None

def title_tokens(title: str):
    toks = [t.lower() for t in TITLE_TOKEN_RE.findall(title)]
    toks = [t for t in toks if len(t) >= 3 and not t.isdigit()]
    return toks

def build_item_table(movies: pd.DataFrame, ratings: pd.DataFrame):
    movies = movies.copy()
    movies["year"] = movies["title"].apply(extract_year).fillna(1995).astype(int)
    movies["genre_list"] = movies["genres"].str.split("|")
    movies["title_tokens"] = movies["title"].apply(title_tokens)

    agg = ratings.groupby("movie_id").agg(
        rating_mean=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    items = movies.merge(agg, on="movie_id", how="left")
    items["rating_mean"] = items["rating_mean"].fillna(items["rating_mean"].mean())
    items["rating_count"] = items["rating_count"].fillna(0).astype(int)
    return items

def build_user_profiles(ratings: pd.DataFrame, items: pd.DataFrame):
    # user-genre affinity from rated items
    it = items[["movie_id", "genre_list"]].explode("genre_list")
    merged = ratings.merge(it, on="movie_id", how="left")

    # weighted by rating
    g = merged.groupby(["user_id", "genre_list"])["rating"].mean().reset_index()
    # normalize per user
    g["genre_affinity"] = g.groupby("user_id")["rating"].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
    prof = g[["user_id", "genre_list", "genre_affinity"]]
    return prof

def sample_query_for_session(user_id: int, user_prof: pd.DataFrame, items: pd.DataFrame, cfg: SessionConfig, rng: np.random.Generator):
    # 70%: genre query, 30%: title-token query
    if rng.random() < 0.7 and (user_prof["user_id"] == user_id).any():
        up = user_prof[user_prof["user_id"] == user_id].sort_values("genre_affinity", ascending=False)
        top_genres = up["genre_list"].head(6).tolist()
        q = rng.choice(top_genres) if len(top_genres) else "Drama"
        query_type = "genre"
        return str(q).lower(), query_type

    # title token query from a popular item
    pop = items.sort_values("rating_count", ascending=False).head(800)
    row = pop.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
    toks = row["title_tokens"][:cfg.max_query_terms_per_item]
    q = rng.choice(toks) if len(toks) else "movie"
    return str(q).lower(), "token"

def retrieve_candidates(query: str, query_type: str, items: pd.DataFrame):
    if query_type == "genre":
        mask = items["genre_list"].apply(lambda gs: query.capitalize() in gs if isinstance(gs, list) else False)
    else:
        mask = items["title_tokens"].apply(lambda ts: query in ts if isinstance(ts, list) else False)

    cand = items[mask].copy()
    return cand

def assign_relevance(user_id: int, movie_id: int, ratings_map: dict, base_pop: float, cfg: SessionConfig, rng: np.random.Generator):
    # Use actual user rating if exists; else approximate with popularity
    r = ratings_map.get((user_id, movie_id), None)
    if r is None:
        # pseudo rating derived from popularity with noise
        pseudo = 2.5 + 1.2 * (base_pop - 0.5) + rng.normal(0, cfg.noise)
        r = float(np.clip(pseudo, 0.5, 5.0))

    # ordinal relevance (0-3)
    if r >= 4.5:
        rel = 3
    elif r >= 3.5:
        rel = 2
    elif r >= 2.5:
        rel = 1
    else:
        rel = 0

    return rel, r

def simulate_click(prob_rel: float, position: int, cfg: SessionConfig, rng: np.random.Generator):
    # position bias: earlier positions more likely
    pos_bias = 1.0 / (position ** cfg.click_position_bias)
    p = np.clip(prob_rel * pos_bias, 0.0, 1.0)
    click = int(rng.random() < p)
    # dwell time in seconds (clicked only)
    dwell = 0.0
    if click:
        dwell = float(np.clip(rng.normal(20 + 60 * prob_rel, 10), 5, 180))
    return click, dwell

def main():
    paths = ensure_dirs()
    cfg = SessionConfig()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    users, movies, ratings = load_movielens_1m(paths.raw)
    items = build_item_table(movies, ratings)
    user_prof = build_user_profiles(ratings, items)

    # for fast lookup
    ratings_map = {(int(u), int(m)): float(r) for u, m, r in ratings[["user_id", "movie_id", "rating"]].itertuples(index=False, name=None)}

    # popularity score in [0,1]
    items["popularity"] = (items["rating_count"] - items["rating_count"].min()) / (items["rating_count"].max() - items["rating_count"].min() + 1e-9)

    sessions = []
    session_id = 0

    user_ids = users["user_id"].unique()
    for user_id in tqdm(user_ids, desc="Simulating sessions"):
        for _ in range(cfg.sessions_per_user):
            query, qtype = sample_query_for_session(int(user_id), user_prof, items, cfg, rng)
            cand = retrieve_candidates(query, qtype, items)

            if len(cand) < cfg.min_candidates:
                # fallback: broaden to top popular items
                cand = items.sort_values("rating_count", ascending=False).head(cfg.candidates_per_session).copy()

            # sample candidates
            cand = cand.sample(
                n=min(cfg.candidates_per_session, len(cand)),
                random_state=int(rng.integers(0, 1_000_000))
            ).reset_index(drop=True)

            # assign rel + behavior with an initial random presentation order
            cand = cand.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

            for i, row in cand.iterrows():
                base_pop = float(row["popularity"])
                rel, used_rating = assign_relevance(int(user_id), int(row["movie_id"]), ratings_map, base_pop, cfg, rng)

                # convert rel to probability for click simulation
                prob_rel = sigmoid((rel - 1.0) * 1.2 + rng.normal(0, cfg.noise))
                click, dwell = simulate_click(prob_rel, position=i + 1, cfg=cfg, rng=rng)

                sessions.append({
                    "session_id": session_id,
                    "user_id": int(user_id),
                    "query": query,
                    "query_type": qtype,
                    "movie_id": int(row["movie_id"]),
                    "title": row["title"],
                    "genres": row["genres"],
                    "year": int(row["year"]),
                    "relevance": int(rel),
                    "observed_rating": float(used_rating),
                    "popularity": float(base_pop),
                    "position_shown": int(i + 1),
                    "clicked": int(click),
                    "dwell_time": float(dwell),
                })

            session_id += 1

    df = pd.DataFrame(sessions)
    out_path = paths.processed / "sessions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote: {out_path} rows={len(df):,} sessions={df['session_id'].nunique():,}")

if __name__ == "__main__":
    main()
