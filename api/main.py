import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import requests
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel


PRECOMPUTE_DIR = os.getenv("PRECOMPUTE_DIR", "outputs")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
POSTER_CACHE_FILE = os.path.join(PRECOMPUTE_DIR, "poster_cache.json")


def _load_poster_cache() -> dict[str, str]:
    try:
        if os.path.exists(POSTER_CACHE_FILE):
            with open(POSTER_CACHE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_poster_cache(cache: dict[str, str]) -> None:
    try:
        os.makedirs(PRECOMPUTE_DIR, exist_ok=True)
        with open(POSTER_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False)
    except Exception:
        # Cache persistence is best effort
        pass


def _load_parquet(path: str) -> pd.DataFrame:
    base = PRECOMPUTE_DIR
    candidates = []
    exact = os.path.join(base, path)
    if os.path.exists(exact):
        candidates.append(exact)
    if not path.endswith(".parquet"):
        with_ext = os.path.join(base, f"{path}.parquet")
        if os.path.exists(with_ext):
            candidates.append(with_ext)
    if not candidates:
        return pd.DataFrame()
    for cand in candidates:
        try:
            return pd.read_parquet(cand, engine="pyarrow")
        except Exception:
            continue
    return pd.DataFrame()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data on startup
    load_data()
    # Pre-compute analytics
    compute_analytics()
    # Pre-compute genres list
    _precompute_genres()
    yield


def _precompute_genres() -> None:
    """Pre-compute and cache genre list at startup."""
    try:
        movies = app.state.movies
        if movies.empty:
            app.state.genres_cache = []
            return
        def split_genres(s: str) -> list[str]:
            if not isinstance(s, str):
                return []
            return [g.strip() for g in s.replace("|", ",").split(",") if g.strip()]
        all_genres: set[str] = set()
        for g in movies["genres"].dropna().tolist():
            for token in split_genres(g):
                all_genres.add(token)
        app.state.genres_cache = sorted(all_genres)
    except Exception:
        app.state.genres_cache = []


app = FastAPI(title="MovieLens Recommender API", version="0.1.0", lifespan=lifespan)
app.mount("/ui", StaticFiles(directory="web", html=True), name="ui")


class Feedback(BaseModel):
    userId: str
    movieId: str
    action: str  # click|like|dismiss


def load_data() -> None:
    app.state.user_topn = _load_parquet("user_topn")
    app.state.popularity = _load_parquet("popularity")
    app.state.movies = _load_parquet("movies_meta")
    app.state.poster_cache = _load_poster_cache()
    factors = _load_parquet("item_factors")
    if not factors.empty:
        # Precompute normalized vectors for cosine
        feats = np.array(factors["features"].tolist(), dtype=float)
        norms = np.linalg.norm(feats, axis=1)
        with np.errstate(invalid="ignore"):
            feats_norm = feats / norms[:, None]
        app.state.item_ids = factors["movieId"].tolist()
        app.state.item_mat = feats_norm
    else:
        app.state.item_ids = []
        app.state.item_mat = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/recommendations/user/{user_id}")
def recs_for_user(
    user_id: str,
    topN: int = 10,
    genres: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[dict]:
    df = app.state.user_topn
    if df.empty:
        return []
    out = df[df.userId == user_id]
    if genres:
        glist = {g.strip() for g in genres.split(",") if g.strip()}
        special = {g.lower() for g in glist}
        wants_none = any(
            t in {"no genre listed", "none", "_none_", "(genres not listed)", "genres not listed", "(no genres listed)"}
            for t in special
        )
        plain = {g for g in glist if g.lower() not in {"no genre listed", "none", "_none_"}}
        mask = False
        if plain:
            mask = out["genres"].fillna("").apply(lambda s: any(g in s for g in plain))
        if wants_none:
            mask = mask | out["genres"].isna() | (out["genres"].astype(str).str.strip() == "")
        out = out[mask]
    if year_from is not None:
        out = out[out["year"].fillna(0) >= year_from]
    if year_to is not None:
        out = out[out["year"].fillna(9999) <= year_to]
    out = out.sort_values("score", ascending=False).head(topN)
    return out.to_dict(orient="records")


@app.get("/recommendations/item/{movie_id}")
def recs_for_item(movie_id: str, topN: int = 10) -> List[dict]:
    ids = app.state.item_ids
    mat = app.state.item_mat
    movies = app.state.movies
    if mat is None or not ids:
        return []
    try:
        idx = ids.index(movie_id)
    except ValueError:
        return []
    target = mat[idx]
    sims = mat @ target
    # Mask self
    sims[idx] = -np.inf
    top_idx = np.argpartition(-sims, range(min(topN, len(sims))))[:topN]
    result = []
    for i in top_idx:
        mid = ids[i]
        score = float(sims[i])
        row = {"movieId": mid, "score": score}
        if not movies.empty:
            mrow = movies[movies.movieId == mid].head(1)
            if not mrow.empty:
                row.update({"title": mrow.iloc[0].title, "genres": mrow.iloc[0].genres, "year": int(mrow.iloc[0].year) if not pd.isna(mrow.iloc[0].year) else None})
        result.append(row)
    result.sort(key=lambda x: x["score"], reverse=True)
    return result[:topN]


@app.get("/popular")
def popular(topN: int = 10, genres: Optional[str] = None) -> List[dict]:
    pop = app.state.popularity
    if pop.empty:
        return []
    df = pop.copy()
    if genres:
        glist = {g.strip() for g in genres.split(",") if g.strip()}
        special = {g.lower() for g in glist}
        wants_none = any(
            t in {"no genre listed", "none", "_none_", "(genres not listed)", "genres not listed", "(no genres listed)"}
            for t in special
        )
        plain = {g for g in glist if g.lower() not in {"no genre listed", "none", "_none_"}}
        mask = False
        if plain:
            mask = df["genres"].fillna("").apply(lambda s: any(g in s for g in plain))
        if wants_none:
            mask = mask | df["genres"].isna() | (df["genres"].astype(str).str.strip() == "")
        df = df[mask]
    else:
        # Aggregate to global popularity by movieId
        df = df.groupby(["movieId", "title", "genres", "year"], as_index=False)["pop_score"].max()
    df = df.sort_values("pop_score", ascending=False).head(topN)
    return df.to_dict(orient="records")


@app.get("/genres")
def list_genres() -> List[str]:
    """Return cached genres list (pre-computed at startup)."""
    cache = getattr(app.state, "genres_cache", None)
    if cache is not None:
        return cache
    # Fallback if not pre-computed
    _precompute_genres()
    return getattr(app.state, "genres_cache", [])


@app.get("/favicon.ico")
def favicon() -> Response:
    # Redirect to our static SVG favicon to avoid 404 noise in browsers
    return RedirectResponse(url="/ui/favicon.svg")


@app.get("/movies")
def browse_movies(topN: int = 50, genres: Optional[str] = None, year_from: Optional[int] = None, year_to: Optional[int] = None, q: Optional[str] = None) -> List[dict]:
    movies = app.state.movies
    if movies.empty:
        return []
    
    # Cache genre-only queries (no year/search filters)
    if genres and not year_from and not year_to and not q and topN in [50, 200]:
        cache_key = (genres, topN)
        cache = getattr(app.state, "genre_query_cache", {})
        now = time.time()
        cached = cache.get(cache_key)
        if cached and (now - cached[0] < 3600):  # 1 hour TTL
            return cached[1]
    
    df = movies.copy()
    if genres:
        glist = {g.strip() for g in genres.split(",") if g.strip()}
        special = {g.lower() for g in glist}
        wants_none = any(
            t in {"no genre listed", "none", "_none_", "(genres not listed)", "genres not listed", "(no genres listed)"}
            for t in special
        )
        plain = {g for g in glist if g.lower() not in {"no genre listed", "none", "_none_", "(genres not listed)", "genres not listed", "(no genres listed)"}}
        mask = False
        if plain:
            mask = df["genres"].fillna("").apply(lambda s: any(g in s for g in plain))
        if wants_none:
            mask = mask | df["genres"].isna() | (df["genres"].astype(str).str.strip() == "") | (df["genres"].astype(str).str.contains(r"\(no genres listed\)", case=False, na=False))
        df = df[mask]
    if year_from is not None:
        df = df[df["year"].fillna(0) >= year_from]
    if year_to is not None:
        df = df[df["year"].fillna(9999) <= year_to]
    if q:
        qlower = q.lower()
        df = df[df["title"].fillna("").str.lower().str.contains(qlower)]
    # If popularity is available, rank by it; else leave arbitrary
    pop = app.state.popularity
    if not pop.empty and "pop_score" in pop.columns:
        pop_small = pop[["movieId", "pop_score"]]
        df = df.merge(pop_small, on="movieId", how="left").sort_values("pop_score", ascending=False)
    
    result = df.head(topN).to_dict(orient="records")
    
    # Cache genre-only queries
    if genres and not year_from and not year_to and not q and topN in [50, 200]:
        cache_key = (genres, topN)
        cache = getattr(app.state, "genre_query_cache", {})
        cache[cache_key] = (time.time(), result)
        setattr(app.state, "genre_query_cache", cache)
    
    return result


@app.get("/posters")
def posters(movieIds: str, size: str = "w342") -> dict:
    ids = [m.strip() for m in movieIds.split(",") if m.strip()]
    movies = app.state.movies
    if not ids or movies.empty:
        return {}

    allowed_sizes = {"w185", "w342", "w500", "original"}
    poster_size = size if size in allowed_sizes else "w342"

    cache: dict[str, str] = getattr(app.state, "poster_cache", None) or {}
    if not cache:
        cache = _load_poster_cache()
        setattr(app.state, "poster_cache", cache)
    out: dict[str, str] = {}

    updated = False

    # If no TMDB key, still serve anything already cached
    if not TMDB_API_KEY:
        for mid in ids[:50]:
            cached = cache.get(mid)
            if not cached:
                continue
            poster_path = None
            if isinstance(cached, str):
                if cached.startswith("http") and "/t/p/" in cached:
                    _, _, tail = cached.partition("/t/p/")
                    _, _, path_tail = tail.partition("/")
                    poster_path = "/" + path_tail if path_tail else None
                elif cached.startswith("/"):
                    poster_path = cached
            if poster_path:
                out[mid] = f"https://image.tmdb.org/t/p/{poster_size}{poster_path}"
        return out
    for mid in ids[:50]:
        cached = cache.get(mid)
        if cached:
            # cached may be a path or a full URL from previous runs
            poster_path = None
            if isinstance(cached, str):
                if cached.startswith("http") and "/t/p/" in cached:
                    _, _, tail = cached.partition("/t/p/")
                    _, _, path_tail = tail.partition("/")
                    poster_path = "/" + path_tail if path_tail else None
                elif cached.startswith("/"):
                    poster_path = cached
            if poster_path:
                out[mid] = f"https://image.tmdb.org/t/p/{poster_size}{poster_path}"
                continue
            else:
                out[mid] = cached
                continue

        row = movies[movies.movieId == mid].head(1)
        if row.empty:
            continue
        title = str(row.iloc[0].title)
        year = row.iloc[0].year
        try:
            params = {"api_key": TMDB_API_KEY, "query": title}
            if pd.notna(year):
                params["year"] = str(int(year))
            r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=3)
            if r.ok:
                js = r.json()
                results = js.get("results") or []
                poster_path = None
                for cand in results:
                    if cand.get("poster_path"):
                        poster_path = cand["poster_path"]
                        break
                if poster_path:
                    cache[mid] = poster_path
                    out[mid] = f"https://image.tmdb.org/t/p/{poster_size}{poster_path}"
                    updated = True
        except Exception:
            continue
    if updated:
        _save_poster_cache(cache)
    return out


@app.get("/movies/by_ids")
def movies_by_ids(movieIds: str) -> List[dict]:
    ids = [m.strip() for m in movieIds.split(",") if m.strip()]
    movies = app.state.movies
    if not ids or movies.empty:
        return []
    df = movies[movies["movieId"].isin(ids)]
    df = df.set_index("movieId").reindex(ids).reset_index()
    return df.to_dict(orient="records")


@app.get("/users")
def list_users(limit: int = 200) -> List[str]:
    """Return a sample of user IDs present in user_topn artifacts."""
    df = app.state.user_topn
    if df.empty or "userId" not in df.columns:
        return []
    users = df["userId"].dropna().astype(str).unique().tolist()
    # Sort numerically when possible, lexicographically otherwise (numeric first)
    def sort_key(u: str):
        s = u.strip()
        if s.isdigit():
            return (0, int(s), s)
        # attempt int-like (e.g., '0012') handled by isdigit(); for mixed tokens keep lexical
        return (1, float('inf'), s)
    users.sort(key=sort_key)
    return users[: max(0, int(limit))]


@app.get("/years")
def years() -> dict:
    """Return min/max year and the list of available years from movies metadata."""
    movies = app.state.movies
    if movies.empty or "year" not in movies.columns:
        return {"min": None, "max": None, "years": []}
    yrs = movies["year"].dropna().astype(int)
    if yrs.empty:
        return {"min": None, "max": None, "years": []}
    vals = sorted(set(int(x) for x in yrs.tolist()))
    return {"min": vals[0], "max": vals[-1], "years": vals}


@app.get("/history")
def history(userId: Optional[str] = None, topN: int = 20) -> List[dict]:
    fb_dir = os.path.join(PRECOMPUTE_DIR, "feedback")
    if not os.path.isdir(fb_dir):
        return []
    files = sorted(
        (os.path.join(fb_dir, f) for f in os.listdir(fb_dir) if f.endswith(".jsonl")),
        reverse=True,
    )
    events: List[dict] = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        if not isinstance(rec, dict):
                            continue
                        if userId and rec.get("userId") != userId:
                            continue
                        events.append(rec)
                    except Exception:
                        continue
        except Exception:
            continue
        if len(events) >= topN:
            break
    events.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return events[:topN]
@app.post("/feedback")
def feedback(fb: Feedback) -> dict:
    out_dir = os.path.join(PRECOMPUTE_DIR, "feedback")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{datetime.utcnow().date().isoformat()}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.utcnow().isoformat(), **fb.model_dump()}, ensure_ascii=False) + "\n")
    return {"status": "ok"}


@app.get("/feedback/summary")
def feedback_summary(
    userId: Optional[str] = None,
    topN: int = 20,
    window_days: int = 30,
    half_life_days: int = 14,
    w_click: float = 0.0,
    w_list: float = 1.0,
    blend_baseline: float = 0.2,
) -> List[dict]:
    fb_dir = os.path.join(PRECOMPUTE_DIR, "feedback")
    pop = app.state.popularity
    # Cache with TTL 60s
    cache_key = (userId or "__all__", int(topN), int(window_days), int(half_life_days), float(w_click), float(w_list), float(blend_baseline))
    now = time.time()
    cache = getattr(app.state, "pop_cache", {})
    cached = cache.get(cache_key)
    if cached and (now - cached[0] < 60):
        return cached[1]

    if not os.path.isdir(fb_dir):
        # Fall back to baseline popularity
        if pop.empty:
            return []
        df = pop.sort_values("pop_score", ascending=False).head(topN)
        result = df[["movieId", "title", "genres", "year", "pop_score"]].to_dict(orient="records")
        cache[cache_key] = (now, result)
        setattr(app.state, "pop_cache", cache)
        return result

    cutoff = datetime.utcnow().date() - timedelta(days=max(0, int(window_days)))
    files = sorted(
        (os.path.join(fb_dir, f) for f in os.listdir(fb_dir) if f.endswith(".jsonl")),
        reverse=True,
    )
    candidates: list[str] = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        if len(name) == 10:
            try:
                fdate = datetime.strptime(name, "%Y-%m-%d").date()
                if fdate >= cutoff:
                    candidates.append(p)
            except Exception:
                continue
    if not candidates:
        candidates = files[:5]

    decay_half = max(1, int(half_life_days))
    weights = {"click": float(w_click), "list": float(w_list)}
    scores: dict[str, float] = {}
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if userId and rec.get("userId") != userId:
                        continue
                    act = rec.get("action")
                    if act not in weights:
                        continue
                    mid = rec.get("movieId")
                    if not isinstance(mid, str):
                        continue
                    # Age days based on timestamp if present
                    age_days = 0.0
                    ts = rec.get("ts")
                    try:
                        if isinstance(ts, str):
                            # tolerate Z suffix and fractional secs
                            tstr = ts.replace("Z", "+00:00").split(".")[0]
                            age_days = (datetime.utcnow() - datetime.fromisoformat(tstr)).total_seconds() / 86400.0
                    except Exception:
                        age_days = 0.0
                    decay = 2 ** (-(age_days / float(decay_half)))
                    scores[mid] = scores.get(mid, 0.0) + weights[act] * decay
        except Exception:
            continue

    # Normalize trending
    trend_norm: dict[str, float]
    if scores:
        vals = list(scores.values())
        vmin, vmax = min(vals), max(vals)
        denom = (vmax - vmin) if vmax > vmin else 1.0
        trend_norm = {k: (v - vmin) / denom for k, v in scores.items()}
    else:
        trend_norm = {}

    # Baseline normalization
    base_norm: dict[str, float] = {}
    if not pop.empty and "pop_score" in pop.columns:
        bmin = float(pop["pop_score"].min())
        bmax = float(pop["pop_score"].max())
        bden = (bmax - bmin) if bmax != bmin else 1.0
        for _, row in pop[["movieId", "pop_score"]].iterrows():
            base_norm[str(row.movieId)] = float((row.pop_score - bmin) / bden)

    beta = max(0.0, min(1.0, float(blend_baseline)))
    combined: dict[str, float] = {}
    keys = set(trend_norm) | set(base_norm)
    if not keys and not pop.empty:
        df = pop.sort_values("pop_score", ascending=False).head(topN)
        result = df[["movieId", "title", "genres", "year", "pop_score"]].to_dict(orient="records")
        cache[cache_key] = (now, result)
        setattr(app.state, "pop_cache", cache)
        return result
    for k in keys:
        t = trend_norm.get(k, 0.0)
        b = base_norm.get(k, 0.0)
        combined[k] = (1.0 - beta) * t + beta * b

    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:topN]
    movie_ids = [mid for mid, _ in top]
    movies = app.state.movies
    lookup = {}
    if not movies.empty:
        meta = movies[movies["movieId"].isin(movie_ids)][["movieId", "title", "genres", "year"]]
        lookup = {row.movieId: {"title": row.title, "genres": row.genres, "year": int(row.year) if not pd.isna(row.year) else None} for _, row in meta.iterrows()}
    result = []
    for mid, score in top:
        row = {"movieId": mid, "score": float(score)}
        if mid in lookup:
            row.update(lookup[mid])
        result.append(row)
    cache[cache_key] = (now, result)
    setattr(app.state, "pop_cache", cache)
    return result


def compute_analytics() -> None:
    """Pre-compute analytics on startup and cache."""
    try:
        movies = app.state.movies
        popularity = app.state.popularity
        # Stable path for ratings.csv regardless of CWD or PRECOMPUTE_DIR
        repo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
        ratings_path = os.path.join(repo_root, "data", "movielens", "32m", "ratings.csv")

        stats = {
            "total_movies": len(movies) if not movies.empty else 0,
            "total_ratings": 0,
            "unique_users": 0,
            "avg_rating": 0.0,
            "top_genres": [],
            "rating_distribution": [],
            "movies_by_year": [],
            "top_rated_movies": [],
            "top_users": [],
            "user_activity_by_month": []
        }

        if not movies.empty and "genres" in movies.columns:
            genre_counts: dict[str, int] = {}
            for genres_str in movies["genres"].dropna():
                for genre in str(genres_str).split("|"):
                    genre = genre.strip()
                    if genre and genre != "(no genres listed)":
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            stats["top_genres"] = [{"genre": g, "count": c} for g, c in top_genres]

        if not movies.empty and "year" in movies.columns:
            year_counts = movies.groupby("year").size().reset_index(name="count")
            year_counts = year_counts.dropna().sort_values("year")
            year_counts = year_counts[year_counts["year"] >= 1900]
            year_counts = year_counts[year_counts["year"] <= 2030]
            stats["movies_by_year"] = [
                {"year": int(row["year"]), "count": int(row["count"])}
                for _, row in year_counts.iterrows()
            ]

        # Load ratings.csv sample to compute rating statistics
        try:
            if os.path.exists(ratings_path):
                # Read a larger but bounded sample with selected columns to improve chart density
                ratings_sample = pd.read_csv(
                    ratings_path,
                    nrows=1000000,
                    usecols=["userId", "movieId", "rating", "timestamp"],
                )
                if not ratings_sample.empty:
                    # Basic stats from ratings
                    stats["total_ratings"] = int(len(ratings_sample))
                    # Ensure userId is treated as string consistently
                    stats["unique_users"] = int(ratings_sample["userId"].astype(str).nunique())
                    
                    if "rating" in ratings_sample.columns:
                        stats["avg_rating"] = float(ratings_sample["rating"].mean())
                        
                        # Rating distribution
                        rating_dist_df = ratings_sample["rating"].value_counts().sort_index().reset_index()
                        rating_dist_df.columns = ["rating", "count"]
                        dist_rows = []
                        for _, row in rating_dist_df.iterrows():
                            rv = float(row["rating"])
                            if np.isnan(rv):
                                rv = 0.0
                            dist_rows.append({"rating": rv, "count": int(row["count"])})
                        stats["rating_distribution"] = dist_rows
                    
                    # Top rated movies by count
                    if "movieId" in ratings_sample.columns:
                        # Coerce types for stability
                        ratings_sample["movieId"] = ratings_sample["movieId"].astype(str)
                        movie_counts = ratings_sample.groupby("movieId", as_index=False).agg(
                            rating_count=("rating", "count"),
                            avg_rating=("rating", "mean")
                        )
                        # Sort by count desc, then by avg_rating desc for tie-breaking
                        top_movies = movie_counts.sort_values(["rating_count", "avg_rating"], ascending=[False, False]).head(20)
                        
                        # Merge with movie titles
                        if not movies.empty and "movieId" in movies.columns:
                            # Ensure consistent type for merge
                            mv = movies.copy()
                            mv["movieId"] = mv["movieId"].astype(str)
                            top_movies = top_movies.merge(mv[["movieId", "title"]], on="movieId", how="left")
                        
                        stats["top_rated_movies"] = [
                            {
                                "title": (row.get("title") if isinstance(row.get("title"), str) and row.get("title") else f"Movie {row.get('movieId', 'Unknown')}"),
                                "rating_count": int(row.get("rating_count", 0)),
                                "avg_rating": float(row.get("avg_rating", 0)),
                            }
                            for _, row in top_movies.iterrows()
                        ]
                    
                    # User-based analytics
                    if "userId" in ratings_sample.columns:
                        ratings_sample["userId"] = ratings_sample["userId"].astype(str)
                        user_rating_counts = ratings_sample["userId"].value_counts().sort_values(ascending=False).head(20)
                        stats["top_users"] = [
                            {"userId": str(uid), "rating_count": int(count)}
                            for uid, count in user_rating_counts.items()
                        ]
                    
                    # User activity over time (if timestamp exists)
                    if "timestamp" in ratings_sample.columns:
                        # Robust datetime conversion; coerce errors to NaT
                        dt = pd.to_datetime(ratings_sample["timestamp"], unit="s", errors="coerce")
                        ratings_sample["date"] = dt
                        # Drop rows with invalid dates to avoid downstream issues
                        valid = ratings_sample.dropna(subset=["date"]).copy()
                        # Use strftime to avoid Pylance warnings about to_period
                        valid["month"] = valid["date"].dt.strftime("%Y-%m")  # type: ignore[attr-defined]
                        monthly_activity = valid.groupby("month", as_index=False).size()
                        monthly_activity.rename(columns={"size": "count"}, inplace=True)
                        # Sort by month ascending and keep last 24
                        monthly_activity = monthly_activity.sort_values("month")
                        tail = monthly_activity.tail(24)
                        stats["user_activity_by_month"] = [
                            {"month": row["month"], "count": int(row["count"])}
                            for _, row in tail.iterrows()
                        ]
        except Exception as e:
            print(f"Error loading ratings: {e}")

        app.state.analytics_cache = stats
    except Exception as exc:
        app.state.analytics_cache = {
            "total_movies": 0,
            "total_ratings": 0,
            "unique_users": 0,
            "avg_rating": 0.0,
            "top_genres": [],
            "rating_distribution": [],
            "movies_by_year": [],
            "top_rated_movies": [],
            "top_users": [],
            "user_activity_by_month": [],
            "error": str(exc),
        }


@app.get("/analytics")
def analytics() -> dict:
    """Return cached dataset statistics and analysis data for the analytics page."""
    cache = getattr(app.state, "analytics_cache", None)
    if cache:
        return cache
    # Fallback if not pre-computed
    compute_analytics()
    return getattr(app.state, "analytics_cache", {})

