import os
import json
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import requests
from pydantic import BaseModel


PRECOMPUTE_DIR = os.getenv("PRECOMPUTE_DIR", "outputs")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


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
    yield


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
        out = out[out["genres"].fillna("").apply(lambda s: any(g in s for g in glist))]
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
        df = df[df["genres"].fillna("").apply(lambda s: any(g in s for g in glist))]
    else:
        # Aggregate to global popularity by movieId
        df = df.groupby(["movieId", "title", "genres", "year"], as_index=False)["pop_score"].max()
    df = df.sort_values("pop_score", ascending=False).head(topN)
    return df.to_dict(orient="records")


@app.get("/genres")
def list_genres() -> List[str]:
    movies = app.state.movies
    if movies.empty:
        return []
    # Split pipe-delimited genres if present, else single token
    def split_genres(s: str) -> list[str]:
        if not isinstance(s, str):
            return []
        return [g.strip() for g in s.replace("|", ",").split(",") if g.strip()]

    all_genres: set[str] = set()
    for g in movies["genres"].dropna().tolist():
        for token in split_genres(g):
            all_genres.add(token)
    return sorted(all_genres)


@app.get("/movies")
def browse_movies(topN: int = 50, genres: Optional[str] = None, year_from: Optional[int] = None, year_to: Optional[int] = None, q: Optional[str] = None) -> List[dict]:
    movies = app.state.movies
    if movies.empty:
        return []
    df = movies.copy()
    if genres:
        glist = {g.strip() for g in genres.split(",") if g.strip()}
        df = df[df["genres"].fillna("").apply(lambda s: any(g in s for g in glist))]
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
    return df.head(topN).to_dict(orient="records")


@app.get("/posters")
def posters(movieIds: str) -> dict:
    ids = [m.strip() for m in movieIds.split(",") if m.strip()]
    movies = app.state.movies
    if not ids or movies.empty or not TMDB_API_KEY:
        return {}
    cache: dict = getattr(app.state, "poster_cache", {})
    setattr(app.state, "poster_cache", cache)
    out: dict[str, str] = {}
    for mid in ids[:100]:
        if mid in cache:
            out[mid] = cache[mid]
            continue
        row = movies[movies.movieId == mid].head(1)
        if row.empty:
            continue
        title = str(row.iloc[0].title)
        year = row.iloc[0].year
        try:
            params = {"api_key": TMDB_API_KEY, "query": title}
            if pd.notna(year):
                params["year"] = int(year)
            r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=10)
            if r.ok:
                js = r.json()
                results = js.get("results") or []
                poster_path = None
                for cand in results:
                    if cand.get("poster_path"):
                        poster_path = cand["poster_path"]
                        break
                if poster_path:
                    url = f"https://image.tmdb.org/t/p/w342{poster_path}"
                    cache[mid] = url
                    out[mid] = url
        except Exception:
            continue
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
def feedback_summary(userId: Optional[str] = None, topN: int = 20, actions: Optional[str] = None) -> List[dict]:
    fb_dir = os.path.join(PRECOMPUTE_DIR, "feedback")
    if not os.path.isdir(fb_dir):
        return []
    counts: dict[str, int] = {}
    files = sorted(
        (os.path.join(fb_dir, f) for f in os.listdir(fb_dir) if f.endswith(".jsonl")),
        reverse=True,
    )
    allowed_actions = {a.strip() for a in (actions.split(",") if actions else ["click", "like"]) if a.strip()}
    for path in files:
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
                    if rec.get("action") not in allowed_actions:
                        continue
                    mid = rec.get("movieId")
                    if not isinstance(mid, str):
                        continue
                    counts[mid] = counts.get(mid, 0) + 1
        except Exception:
            continue
        if len(counts) >= topN * 3:
            break
    if not counts:
        return []
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:topN]
    movie_ids = [mid for mid, _ in items]
    movies = app.state.movies
    lookup = {}
    if not movies.empty:
        meta = movies[movies["movieId"].isin(movie_ids)][["movieId", "title", "genres", "year"]]
        lookup = {row.movieId: {"title": row.title, "genres": row.genres, "year": int(row.year) if not pd.isna(row.year) else None} for _, row in meta.iterrows()}
    result = []
    for mid, cnt in items:
        row = {"movieId": mid, "count": int(cnt)}
        if mid in lookup:
            row.update(lookup[mid])
        result.append(row)
    return result

