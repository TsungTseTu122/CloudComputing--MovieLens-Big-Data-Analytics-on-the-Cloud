from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


def _write_parquet(df: pd.DataFrame, base: Path, name: str) -> None:
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)


def test_api_endpoints_with_fixtures(tmp_path: Path, monkeypatch) -> None:
    # Prepare minimal fixtures
    base = tmp_path / "outputs"
    base.mkdir(parents=True, exist_ok=True)

    movies_meta = pd.DataFrame(
        {"movieId": ["m1", "m2"], "title": ["A (2000)", "B (2001)"], "genres": ["Drama", "Action"], "year": [2000, 2001]}
    )
    user_topn = pd.DataFrame(
        {
            "userId": ["u1", "u1"],
            "movieId": ["m1", "m2"],
            "score": [5.0, 4.0],
            "title": ["A (2000)", "B (2001)"],
            "genres": ["Drama", "Action"],
            "year": [2000, 2001],
        }
    )
    popularity = pd.DataFrame(
        {"movieId": ["m1", "m2"], "pop_score": [100, 50], "title": ["A (2000)", "B (2001)"], "genres": ["Drama", "Action"], "year": [2000, 2001]}
    )
    item_factors = pd.DataFrame({"movieId": ["m1", "m2"], "features": [np.array([1.0, 0.0]), np.array([0.0, 1.0])]})

    _write_parquet(movies_meta, base, "movies_meta")
    _write_parquet(user_topn, base, "user_topn")
    _write_parquet(popularity, base, "popularity")
    _write_parquet(item_factors, base, "item_factors")

    # Point API to fixtures dir
    monkeypatch.setenv("PRECOMPUTE_DIR", str(base))

    # Import after setting env so startup loads fixtures
    from api.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json().get("status") == "ok"

        r = client.get("/recommendations/user/u1", params={"topN": 1})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list) and len(data) == 1
        assert data[0]["movieId"] in {"m1", "m2"}

        r = client.get("/recommendations/item/m1", params={"topN": 1})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list) and len(data) == 1
        assert data[0]["movieId"] == "m2"  # orthogonal vector most similar after masking self

        r = client.get("/popular", params={"topN": 1, "genres": "Drama"})
        assert r.status_code == 200
        data = r.json()
        assert data and data[0]["movieId"] == "m1"

        r = client.post("/feedback", json={"userId": "u1", "movieId": "m1", "action": "click"})
        assert r.status_code == 200 and r.json().get("status") == "ok"

