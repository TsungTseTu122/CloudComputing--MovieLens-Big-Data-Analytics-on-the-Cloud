"""Spark-based collaborative filtering recommender utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import math

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import SQLTransformer, StringIndexer, StringIndexerModel
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

DEFAULT_RATINGS_PATH = "hdfs://namenode:8020/user/hadoop/movielens/ratings.csv"
DEFAULT_MOVIES_PATH = "hdfs://namenode:8020/user/hadoop/movielens/movies.csv"


@dataclass
class TrainingResult:
    """Container for the trained pipeline and evaluation metrics."""

    pipeline_model: PipelineModel
    als_model: ALSModel
    metrics: Dict[str, float]
    test_predictions: DataFrame


def build_spark_session(app_name: str = "MovieLensRecommender", master: Optional[str] = None) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    return builder.getOrCreate()


def load_ratings(spark: SparkSession, path: str) -> DataFrame:
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(path)
        .select(
            F.col("userId").cast(StringType()).alias("userId"),
            F.col("movieId").cast(StringType()).alias("movieId"),
            F.col("rating").cast(DoubleType()).alias("rating"),
            F.col("timestamp"),
        )
        .dropna(subset=["userId", "movieId", "rating"])
    )
    return df


def load_movies(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(path)
        .select(
            F.col("movieId").cast(StringType()).alias("movieId"),
            F.col("title"),
            F.col("genres"),
        )
    )


def build_training_pipeline(rank: int, reg_param: float, max_iter: int, implicit_prefs: bool) -> Pipeline:
    user_indexer = StringIndexer(inputCol="userId", outputCol="userIndex", handleInvalid="skip")
    item_indexer = StringIndexer(inputCol="movieId", outputCol="movieIndex", handleInvalid="skip")
    cast_transformer = SQLTransformer(
        statement="SELECT *, CAST(userIndex AS INT) AS userIndexInt, CAST(movieIndex AS INT) AS movieIndexInt FROM __THIS__"
    )
    als = ALS(
        userCol="userIndexInt",
        itemCol="movieIndexInt",
        ratingCol="rating",
        rank=rank,
        regParam=reg_param,
        maxIter=max_iter,
        implicitPrefs=implicit_prefs,
        nonnegative=True,
        coldStartStrategy="drop",
    )
    return Pipeline(stages=[user_indexer, item_indexer, cast_transformer, als])


def train_model(
    ratings: DataFrame,
    rank: int = 20,
    reg_param: float = 0.1,
    max_iter: int = 15,
    implicit_prefs: bool = False,
) -> TrainingResult:
    training, test = ratings.randomSplit([0.8, 0.2], seed=42)

    pipeline = build_training_pipeline(rank, reg_param, max_iter, implicit_prefs)
    pipeline_model = pipeline.fit(training)

    predictions = pipeline_model.transform(test)
    evaluator_rmse = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="mae")
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    als_model = pipeline_model.stages[-1]
    metrics = {"rmse": rmse, "mae": mae}

    return TrainingResult(
        pipeline_model=pipeline_model,
        als_model=als_model,
        metrics=metrics,
        test_predictions=predictions.select("userId", "movieId", "rating", "prediction"),
    )


def _index_lookup(labels: Iterable[str]) -> Dict[int, str]:
    return {idx: label for idx, label in enumerate(labels)}


def _prepare_user_subset(training_result: TrainingResult, user_id: str) -> DataFrame:
    user_indexer: StringIndexerModel = training_result.pipeline_model.stages[0]  # type: ignore[assignment]
    spark = training_result.test_predictions.sparkSession
    subset = spark.createDataFrame([(user_id,)], ["userId"])
    subset = user_indexer.transform(subset)
    subset = subset.dropna(subset=["userIndex"])
    if subset.count() == 0:
        raise ValueError(f"User {user_id} not found in training data.")
    return subset.withColumn("userIndexInt", F.col("userIndex").cast("int"))


def _prepare_item_subset(training_result: TrainingResult, movie_id: str) -> DataFrame:
    item_indexer: StringIndexerModel = training_result.pipeline_model.stages[1]  # type: ignore[assignment]
    spark = training_result.test_predictions.sparkSession
    subset = spark.createDataFrame([(movie_id,)], ["movieId"])
    subset = item_indexer.transform(subset)
    subset = subset.dropna(subset=["movieIndex"])
    if subset.count() == 0:
        raise ValueError(f"Movie {movie_id} not found in training data.")
    return subset.withColumn("movieIndexInt", F.col("movieIndex").cast("int"))


def _labels_dataframe(spark: SparkSession, item_labels: Dict[int, str]) -> DataFrame:
    data = [(int(idx), mid) for idx, mid in item_labels.items()]
    return spark.createDataFrame(data, ["movieIndexInt", "movieId"])  # type: ignore[arg-type]


def _explode_recommendations(
    recommendations: DataFrame,
    item_labels: Dict[int, str],
    id_column: str,
) -> DataFrame:
    spark = recommendations.sparkSession
    element_type = recommendations.schema["recommendations"].dataType.elementType
    field_names = [name for name in element_type.fieldNames() if name != "rating"]
    if not field_names:
        raise ValueError("Recommendation struct missing index field")
    index_field = field_names[0]

    labels_df = _labels_dataframe(spark, item_labels)

    exploded = (
        recommendations.select(id_column, F.explode("recommendations").alias("rec"))
        .select(
            id_column,
            F.col("rec.rating").alias("score"),
            F.col(f"rec.{index_field}").cast("int").alias("movieIndexInt"),
        )
        .join(labels_df, on="movieIndexInt", how="inner")
    )
    return exploded


def recommend_for_user(
    training_result: TrainingResult,
    user_id: str,
    top_n: int = 10,
    movies: Optional[DataFrame] = None,
) -> DataFrame:
    subset = _prepare_user_subset(training_result, user_id)
    als_model = training_result.als_model

    recommendations = als_model.recommendForUserSubset(subset.select("userIndexInt"), top_n)
    recommendations = recommendations.join(subset, on="userIndexInt", how="inner")

    item_indexer: StringIndexerModel = training_result.pipeline_model.stages[1]  # type: ignore[assignment]
    item_labels = _index_lookup(item_indexer.labels)
    rec_df = _explode_recommendations(recommendations, item_labels, "userIndexInt")
    rec_df = rec_df.join(subset.select("userIndexInt", "userId"), on="userIndexInt", how="inner")
    if movies is not None:
        rec_df = rec_df.join(movies, on="movieId", how="left")
    return rec_df.select("userId", "movieId", "title", "genres", "score") if "title" in rec_df.columns else rec_df.select(
        "userId", "movieId", "score"
    )


def similar_items(
    training_result: TrainingResult,
    movie_id: str,
    top_n: int = 10,
    movies: Optional[DataFrame] = None,
) -> DataFrame:
    item_indexer: StringIndexerModel = training_result.pipeline_model.stages[1]  # type: ignore[assignment]
    label_to_index = {label: idx for idx, label in enumerate(item_indexer.labels)}
    if movie_id not in label_to_index:
        raise ValueError(f"Movie {movie_id} not found in training data.")

    target_index = label_to_index[movie_id]
    spark = training_result.test_predictions.sparkSession
    item_factors = training_result.als_model.itemFactors
    target_row = item_factors.filter(F.col("id") == target_index).select("features").collect()
    if not target_row:
        raise ValueError(f"No factors found for movie {movie_id}")

    target_vector = target_row[0][0]
    norm_target = math.sqrt(sum(value * value for value in target_vector))
    target_lit = F.array(*[F.lit(float(v)) for v in target_vector])

    # Compute cosine similarity without Python UDFs
    dot = F.aggregate(
        F.zip_with(F.col("features"), target_lit, lambda x, y: x * y),
        F.lit(0.0),
        lambda acc, v: acc + v,
    )
    norm_vec = F.sqrt(
        F.aggregate(F.transform(F.col("features"), lambda x: x * x), F.lit(0.0), lambda acc, v: acc + v)
    )
    score_col = F.when((F.lit(norm_target) == 0) | (norm_vec == 0), F.lit(0.0)).otherwise(
        dot / (F.lit(norm_target) * norm_vec)
    )

    scored = (
        item_factors.withColumnRenamed("id", "movieIndexInt")
        .withColumn("score", score_col)
        .filter(F.col("movieIndexInt") != target_index)
        .orderBy(F.col("score").desc())
        .limit(top_n)
    )

    item_labels = _index_lookup(item_indexer.labels)
    labels_df = _labels_dataframe(spark, item_labels)
    result = scored.join(labels_df, on="movieIndexInt", how="inner")
    if movies is not None:
        result = result.join(movies, on="movieId", how="left")
    return result.select("movieId", "title", "genres", "score") if "title" in result.columns else result.select(
        "movieId", "score"
    )


def _user_labels_dataframe(spark: SparkSession, user_labels: Dict[int, str]) -> DataFrame:
    data = [(int(idx), uid) for idx, uid in user_labels.items()]
    return spark.createDataFrame(data, ["userIndexInt", "userId"])  # type: ignore[arg-type]


def _parse_year_from_title(title_col: F.Column) -> F.Column:
    return F.regexp_extract(title_col, r"\((\d{4})\)$", 1).cast("int")


def _write_artifacts(
    tr: TrainingResult,
    ratings: DataFrame,
    movies: Optional[DataFrame],
    out_dir: str,
    topk: int,
) -> None:
    spark = ratings.sparkSession
    # Movies meta with year
    movies_df = movies if movies is not None else spark.createDataFrame([], "movieId string, title string, genres string")
    movies_meta = movies_df.select(
        F.col("movieId"), F.col("title"), F.col("genres"), _parse_year_from_title(F.col("title")).alias("year")
    )
    movies_meta.coalesce(1).write.mode("overwrite").parquet(f"{out_dir}/movies_meta")

    # user_topn
    user_indexer: StringIndexerModel = tr.pipeline_model.stages[0]  # type: ignore[assignment]
    item_indexer: StringIndexerModel = tr.pipeline_model.stages[1]  # type: ignore[assignment]
    user_labels = _index_lookup(user_indexer.labels)
    item_labels = _index_lookup(item_indexer.labels)
    users_df = _user_labels_dataframe(spark, user_labels)
    items_df = _labels_dataframe(spark, item_labels)

    recs_all = tr.als_model.recommendForAllUsers(topk)
    # Normalize the user column name to userIndexInt
    if "userIndexInt" not in recs_all.columns:
        if "userCol" in recs_all.columns:
            recs_all = recs_all.withColumnRenamed("userCol", "userIndexInt")
        else:
            for c in ["user", "userId"]:
                if c in recs_all.columns:
                    recs_all = recs_all.withColumnRenamed(c, "userIndexInt")
                    break

    # Determine the index field inside the recommendations struct (exclude rating)
    rec_elem = recs_all.schema["recommendations"].dataType.elementType
    idx_fields = [fn for fn in rec_elem.fieldNames() if fn != "rating"]
    rec_index_field = idx_fields[0] if idx_fields else "movieIndexInt"

    exploded = (
        recs_all.select(F.col("userIndexInt"), F.explode("recommendations").alias("rec"))
        .select(
            F.col("userIndexInt"),
            F.col("rec.rating").alias("score"),
            F.col("rec").getField(rec_index_field).cast("int").alias("movieIndexInt"),
        )
        .join(users_df, on="userIndexInt", how="inner")
        .join(items_df, on="movieIndexInt", how="inner")
        .join(movies_meta, on="movieId", how="left")
        .select("userId", "movieId", "score", "title", "genres", "year")
    )
    exploded.coalesce(1).write.mode("overwrite").parquet(f"{out_dir}/user_topn")

    # popularity (global)
    pop = (
        ratings.groupBy("movieId").agg(F.count(F.lit(1)).alias("cnt"), F.avg("rating").alias("avg"))
        .withColumn("pop_score", F.col("cnt"))
        .join(movies_meta, on="movieId", how="left")
        .select("movieId", "pop_score", "title", "genres", "year")
    )
    pop.coalesce(1).write.mode("overwrite").parquet(f"{out_dir}/popularity")

    # item_factors with movieId
    item_factors = tr.als_model.itemFactors.withColumnRenamed("id", "movieIndexInt")
    item_factors = item_factors.join(items_df, on="movieIndexInt", how="inner").select("movieId", "features")
    item_factors.coalesce(1).write.mode("overwrite").parquet(f"{out_dir}/item_factors")


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and query the MovieLens ALS recommender.")
    parser.add_argument("--ratings-path", default=DEFAULT_RATINGS_PATH, help="Path to ratings CSV")
    parser.add_argument("--movies-path", default=DEFAULT_MOVIES_PATH, help="Path to movies CSV")
    parser.add_argument("--master", default=None, help="Spark master URL (defaults to cluster config)")
    parser.add_argument("--rank", type=int, default=20, help="Factorization rank")
    parser.add_argument("--reg", type=float, default=0.1, help="ALS regularization parameter")
    parser.add_argument("--max-iter", type=int, default=15, help="Number of ALS iterations")
    parser.add_argument("--implicit", action="store_true", help="Use implicit feedback mode")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to produce")
    parser.add_argument("--user-id", help="User ID for which to produce recommendations")
    parser.add_argument("--movie-id", help="Movie ID for which to produce similar-item suggestions")
    parser.add_argument("--output", help="Optional path to write recommendations (Parquet)")
    parser.add_argument("--master-local", action="store_true", help="Force local[*] master for testing")
    # Artifact writing
    parser.add_argument("--write-artifacts", action="store_true", help="Write precomputed artifacts and exit")
    parser.add_argument("--precompute-dir", default="outputs", help="Directory to write artifacts")
    parser.add_argument("--topn-k", type=int, default=100, help="Top-K size for precomputed recommendations")
    return parser.parse_args(list(args) if args is not None else None)


def main(cli_args: Optional[Iterable[str]] = None) -> int:
    args = parse_args(cli_args)
    master = "local[*]" if args.master_local else args.master
    spark = build_spark_session(master=master)

    ratings = load_ratings(spark, args.ratings_path)
    if ratings.count() == 0:
        raise SystemExit("Ratings dataset is empty")

    movies = None
    try:
        movies = load_movies(spark, args.movies_path)
    except Exception:  # pragma: no cover - dataset optional
        movies = None

    training_result = train_model(
        ratings,
        rank=args.rank,
        reg_param=args.reg,
        max_iter=args.max_iter,
        implicit_prefs=args.implicit,
    )

    # Optionally write precomputed artifacts and exit
    if args.write_artifacts:
        _write_artifacts(
            training_result,
            ratings=ratings,
            movies=movies,
            out_dir=args.precompute_dir,
            topk=args.topn_k,
        )
        print(f"Artifacts written under {args.precompute_dir}")
        spark.stop()
        return 0

    print("Evaluation metrics:")
    for metric, value in training_result.metrics.items():
        print(f"  {metric}: {value:.4f}")

    output_df: Optional[DataFrame] = None

    if args.user_id:
        recs = recommend_for_user(training_result, args.user_id, args.top_n, movies)
        recs = recs.orderBy(F.col("score").desc())
        recs.show(truncate=False)
        output_df = recs

    if args.movie_id:
        similar = similar_items(training_result, args.movie_id, args.top_n, movies)
        similar = similar.orderBy(F.col("score").desc())
        similar.show(truncate=False)
        output_df = similar

    if args.output and output_df is not None:
        output_df.write.mode("overwrite").parquet(args.output)
        print(f"Recommendations written to {args.output}")

    spark.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
