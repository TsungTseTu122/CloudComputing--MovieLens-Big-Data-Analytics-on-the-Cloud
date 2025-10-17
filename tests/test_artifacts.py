from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession

from src import recommendation as rec


def test_write_artifacts_smoke(tmp_path: Path) -> None:
    spark = rec.build_spark_session(app_name="test-artifacts", master="local[1]")
    try:
        ratings = spark.createDataFrame(
            [
                ("1", "10", 4.0, 1000),
                ("1", "11", 5.0, 1001),
                ("2", "10", 3.0, 1002),
                ("2", "12", 4.0, 1003),
                ("3", "11", 2.0, 1004),
                ("3", "12", 4.5, 1005),
            ],
            ["userId", "movieId", "rating", "timestamp"],
        )

        movies = spark.createDataFrame(
            [
                ("10", "Movie A (2000)", "Action"),
                ("11", "Movie B (2001)", "Drama"),
                ("12", "Movie C (2002)", "Comedy"),
            ],
            ["movieId", "title", "genres"],
        )

        tr = rec.train_model(ratings, rank=4, reg_param=0.05, max_iter=5)

        out_dir = tmp_path / "outputs"
        rec._write_artifacts(tr, ratings=ratings, movies=movies, out_dir=str(out_dir), topk=5)  # type: ignore[attr-defined]

        # Assert Parquet folders exist and are non-empty
        for sub in ["user_topn", "item_factors", "popularity", "movies_meta"]:
            p = out_dir / sub
            assert p.exists(), f"missing artifact {sub}"
            assert any(p.glob("*.parquet")) or any(p.glob("*/*")), f"empty artifact {sub}"
    finally:
        spark.stop()

