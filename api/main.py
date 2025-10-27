import os
import json
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel


PRECOMPUTE_DIR = os.getenv("PRECOMPUTE_DIR", "outputs")


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


app = FastAPI(title="MovieLens Recommender API", version="0.1.0")


class Feedback(BaseModel):
    userId: str
    movieId: str
    action: str  # click|like|dismiss


@app.on_event("startup")
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


@app.post("/feedback")
def feedback(fb: Feedback) -> dict:
    out_dir = os.path.join(PRECOMPUTE_DIR, "feedback")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{datetime.utcnow().date().isoformat()}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.utcnow().isoformat(), **fb.dict()}, ensure_ascii=False) + "\n")
    return {"status": "ok"}

