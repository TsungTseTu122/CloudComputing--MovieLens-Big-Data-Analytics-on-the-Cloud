"""Smoke tests for the recommender pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import recommendation


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = recommendation.build_spark_session(app_name="test-recommender", master="local[1]")
    yield spark
    spark.stop()


def test_train_and_recommend(spark: SparkSession) -> None:
    ratings = spark.createDataFrame(
        [
            ("1", "10", 4.0, 1000),
            ("1", "11", 5.0, 1001),
            ("2", "10", 3.0, 1002),
            ("2", "12", 4.0, 1003),
            ("3", "11", 2.0, 1004),
            ("3", "12", 4.5, 1005),
            ("4", "10", 4.5, 1006),
            ("4", "13", 3.5, 1007),
            ("5", "13", 4.0, 1008),
            ("5", "11", 3.5, 1009),
        ],
        ["userId", "movieId", "rating", "timestamp"],
    )

    result = recommendation.train_model(ratings, rank=4, reg_param=0.05, max_iter=5)

    assert {"rmse", "mae"} <= set(result.metrics)

    movies = spark.createDataFrame(
        [
            ("10", "Movie A", "Action"),
            ("11", "Movie B", "Drama"),
            ("12", "Movie C", "Comedy"),
            ("13", "Movie D", "Sci-Fi"),
        ],
        ["movieId", "title", "genres"],
    )

    user_recs = recommendation.recommend_for_user(result, "1", top_n=2, movies=movies)
    assert user_recs.count() > 0
    assert "title" in user_recs.columns

    item_recs = recommendation.similar_items(result, "10", top_n=2, movies=movies)
    assert item_recs.count() > 0
