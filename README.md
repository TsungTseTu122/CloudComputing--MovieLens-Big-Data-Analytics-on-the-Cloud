# CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud

## Demo

Watch a short walkthrough of the UI and API:

- Demo video: docs/demo.mp4 (replace this link with your hosted video or GIF)
- Quick highlights:
  - Dataset switch to MovieLens 32M with one command
  - API + Swagger and a Netflix‑style web UI at `/ui`
  - Polished user picker, genre tags, year presets, and instant rendering


## Overview
This project analyzes the MovieLens dataset using PySpark, Hadoop HDFS, and Docker to perform clustering, classification, and association rule mining on user-movie interactions. The system runs in a containerized cloud environment with Spark clusters, enabling scalable big data processing.

## Project Structure
CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud
```
│── README.md                         # Project overview & setup guide
│── requirements.txt                  # Python dependencies (PySpark, tooling)
│── docker-compose.yml                # Docker setup for Hadoop, Spark, and Jupyter
│
├── data/                             # MovieLens artifacts downloaded by the helper script
│   ├── MovieLens_Dataset_Info.txt    # Dataset license and source information
│   └── movielens/
│       └── 32m/                      # Example extracted dataset variant
│           ├── ratings.csv
│           ├── movies.csv
│           ├── tags.csv
│           └── links.csv
│
├── notebooks/
│   ├── Project.ipynb                 # Original exploratory analysis
│   └── Recommender.ipynb             # ALS walkthrough & evaluation examples
│
├── reports/
│   └── TsungTse_Tu_s4780187_proposal.docx
│
├── scripts/
│   ├── download_movielens.py         # Automated dataset downloader & checksum validation
│   └── load_to_hdfs.sh               # Helper to upload CSVs to HDFS (use WSL/Git Bash on Windows)
│
├── src/
│   ├── __init__.py
│   └── recommendation.py             # PySpark ALS training & inference utilities
│
└── tests/
    └── test_recommendation.py        # Smoke tests for the recommender pipeline
```

## Setup and Installation

1. Clone repository

`
git clone https://github.com/TsungTseTu122/CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud.git
`

`
cd CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud
`

2. Set Up Docker Environment
   
Ensure Docker is installed and running. Then, start the containerized services:

`
docker-compose up -d
`

This command activates Hadoop HDFS and Apache Spark services for data processing

3. Access Jupyter Notebook
   
If you are using Jupyter Notebook within Docker, you can access it at:

`
http://localhost:8888/
`

Otherwise, you can run the `.ipynb` file in any local or cloud-based Jupyter environment.

4. Confirm your local clone is connected to GitHub

If you are working across multiple Windows profiles and want to verify where your
changes live, list the configured remotes:

```
git remote -v
```

If no remotes are returned, the work is only stored locally. Link the clone to
your GitHub fork (replace the URL with your repository if different) and push
the branch that contains your latest commits:

```
git remote add origin https://github.com/TsungTseTu122/CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud.git
git push -u origin work
```

After pushing, confirm the branch or pull request on GitHub to ensure the
changes are available remotely.

## Dataset

### 1. Download MovieLens automatically

Run the helper script to download and extract the dataset variant you need (32M by default):

```
python scripts/download_movielens.py --variant 32m
```

The command stores the CSV files under `data/movielens/<variant>/` and records a checksum so repeat runs skip the download unless you add `--force`.

### 2. Load data into HDFS

After the CSVs are available locally, push them into HDFS so Spark can read them inside the cluster:

```
bash scripts/load_to_hdfs.sh
```

The script uploads every CSV in `data/movielens/32m` to `hdfs://namenode:8020/user/hadoop/movielens/`. Override the source directory or container name with the `DATA_DIR`, `HDFS_TARGET`, and `HDFS_CONTAINER` environment variables if your setup differs (for example on Windows with WSL you can run `wsl bash scripts/load_to_hdfs.sh`).

## Key Features

- **Automated dataset ingestion** – Scripts to download MovieLens variants and push them into HDFS with a single command.
- **Collaborative filtering recommender** – A PySpark ALS pipeline that trains on the ratings data, reports RMSE/MAE, and serves top-N user or item recommendations via CLI or notebooks.
- **Exploratory notebooks** – Existing notebooks for clustering, classification, and association analysis, now complemented by a recommender walkthrough.
- **Containerized deployment** – Docker Compose stack for Hadoop and Spark to reproduce the environment quickly.

## Training the recommender

Once the data is in HDFS, you can train and query the recommender directly from the command line:

```
python -m src.recommendation --top-n 10 --user-id 123 --master-local
```

Key options:

- `--ratings-path` / `--movies-path` to override the default HDFS locations.
- `--user-id` to print personalized movie recommendations.
- `--movie-id` to find similar titles.
- `--output` to persist recommendations as Parquet.
- `--master-local` to run a quick local test outside the cluster.

You can also import `src.recommendation` inside a notebook to visualize the results.

## Usage (Cluster Mode)

After uploading CSVs to HDFS, you can run the recommender on the Spark cluster.

1) Ensure services are running

```
docker compose up -d
```

2) Run the recommender on the cluster with tuned settings (example)

```
docker compose exec \
  -e SPARK_HOME=/home/jovyan/spark-3.5.6-bin-hadoop3 \
  -e PYSPARK_SUBMIT_ARGS="--conf spark.executor.instances=1 --conf spark.executor.cores=2 --conf spark.executor.memory=4g --conf spark.executor.memoryOverhead=1g --conf spark.driver.memory=4g --conf spark.sql.shuffle.partitions=64 pyspark-shell" \
  -w /home/jovyan/work \
  jupyter \
  python -m src.recommendation \
    --master spark://spark-master:7077 \
    --ratings-path hdfs://namenode:8020/user/hadoop/movielens/ratings.csv \
    --movies-path  hdfs://namenode:8020/user/hadoop/movielens/movies.csv \
    --rank 10 --max-iter 5 \
    --user-id 1 --top-n 5
```

Example output (truncated):

```
+------+-------+---------------------------------------+------------------+---------+
|userId|movieId|title                                  |genres            |score    |
+------+-------+---------------------------------------+------------------+---------+
|1     |194434 |Adrenaline (1990)                      |(no genres listed)|5.5554385|
|1     |194334 |Les Luthiers: El Grosso Concerto (2001)|(no genres listed)|5.4845195|
|1     |203882 |Dead in the Water (2006)               |Horror            |5.4709344|
|1     |203633 |The Bribe (2018)                       |Comedy|Crime      |5.402133 |
|1     |183947 |NOFX Backstage Passport 2              |(no genres listed)|5.3676186|
+------+-------+---------------------------------------+------------------+---------+
```

Notes:
- The example settings trade some accuracy for stability on a single-node worker. Increase memory/cores based on your machine.
- UIs: Spark Master http://localhost:8080, NameNode http://localhost:9870, DataNode http://localhost:9864, Jupyter http://localhost:8888

## Precompute + Serve (API)

Phase 1 exposes a lightweight API on top of precomputed artifacts.

1) Precompute artifacts (writes to `outputs/`)

Cluster example (tuned resources):

```
docker compose exec \
  -e SPARK_HOME=/home/jovyan/spark-3.5.6-bin-hadoop3 \
  -e PYSPARK_SUBMIT_ARGS="--conf spark.executor.instances=1 --conf spark.executor.cores=2 --conf spark.executor.memory=4g --conf spark.executor.memoryOverhead=1g --conf spark.driver.memory=4g --conf spark.sql.shuffle.partitions=64 pyspark-shell" \
  -w /home/jovyan/work \
  jupyter \
  python -m src.recommendation \
    --master spark://spark-master:7077 \
    --ratings-path hdfs://namenode:8020/user/hadoop/movielens/ratings.csv \
    --movies-path  hdfs://namenode:8020/user/hadoop/movielens/movies.csv \
    --rank 10 --max-iter 5 \
    --write-artifacts --precompute-dir outputs --topn-k 100
```

2) Start the API

```
docker compose up -d api
```

3) Try endpoints

```
curl "http://localhost:8000/health"
curl "http://localhost:8000/recommendations/user/1?topN=5"
curl "http://localhost:8000/recommendations/item/194434?topN=5"
curl "http://localhost:8000/popular?topN=10&genres=Horror"
curl -X POST "http://localhost:8000/feedback" -H "Content-Type: application/json" -d '{"userId":"1","movieId":"194434","action":"click"}'
```

### Web UI (Netflix-style)

Serve a simple, user‑friendly UI at `/ui`:

```
# Ensure artifacts exist under outputs/ and are referenced by the API
$env:PRECOMPUTE_DIR = (Resolve-Path .\outputs).Path    # PowerShell
python -m uvicorn api.main:app --reload --port 8000
```

Then open http://127.0.0.1:8000/ui

What you get:
- Top 50 all‑time (from `/popular`)
- Genre chips with top picks (from `/genres` + `/popular?genres=...`)
- Personalized row: enter `userId` and optional filters to fetch `/recommendations/user/{user_id}`

Posters (optional): set a TMDB API key to display thumbnails

```
# PowerShell
$env:TMDB_API_KEY = "<your_tmdb_key>"
# Bash
export TMDB_API_KEY="<your_tmdb_key>"
```
The UI will call `/posters?movieIds=...` which looks up posters in TMDB by title/year and returns thumbnail URLs.

## Usage (Local Mode)

Run the pipeline in a single JVM (no executors):

```
docker compose exec -w /home/jovyan/work jupyter \
  python -m src.recommendation \
  --master-local \
  --ratings-path hdfs://namenode:8020/user/hadoop/movielens/ratings.csv \
  --movies-path  hdfs://namenode:8020/user/hadoop/movielens/movies.csv \
  --user-id 1 --top-n 5
```

## Notebooks

Open Jupyter at http://localhost:8888 and use the notebooks in `notebooks/`.

### Project.ipynb (Exploration)

Purpose: original exploratory analysis of the MovieLens CSVs (schema, sampling, quick visuals). Loads the MovieLens 32M files from `data/movielens/32m` (auto-downloads via `scripts/download_movielens.py` if missing) and walks through the cleaning and analysis steps in Jupyter—no Hadoop/HDFS setup required.

Starter cell (local):

```
from pyspark.sql import SparkSession

movies_path = "data/movielens/32m/movies.csv"

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("MovieLensAnalysis")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

movies = spark.read.csv(movies_path, header=True, inferSchema=True)
movies.show(20, truncate=False)
```

Manual steps to run in Jupyter:
1) Install deps: `pip install -r requirements.txt`.
2) Download data once: `python scripts/download_movielens.py --variant 32m` (stores under `data/movielens/32m`).
3) Start Jupyter: `jupyter notebook` and open `notebooks/Project.ipynb`.
4) Run cells top-to-bottom; the setup cell will reuse the local CSVs. Use Spark cells for the full flow or the pandas preview cell for a quick look.

### Recommender.ipynb (ALS Walkthrough)

Purpose: train/evaluate ALS and display top‑N recommendations.

Starter cell for the cluster session:

```
from pyspark.sql import SparkSession
spark = (
    SparkSession.builder
    .master("spark://spark-master:7077")
    .appName("RecommenderNotebook")
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)

from src import recommendation as rec
ratings = rec.load_ratings(spark, "hdfs://namenode:8020/user/hadoop/movielens/ratings.csv")
movies  = rec.load_movies(spark,  "hdfs://namenode:8020/user/hadoop/movielens/movies.csv")
res = rec.train_model(ratings, rank=10, max_iter=5, reg_param=0.1)
rec.recommend_for_user(res, "1", top_n=5, movies=movies).show(truncate=False)
```

## Report

- File: `reports/TsungTse_Tu_s4780187_proposal.docx`
- Purpose: original project proposal outlining goals, methods (clustering, classification, association rules), and data sources.
- How to open: Microsoft Word, LibreOffice, or Google Docs.
- Note: for reference only — not required to run the stack.

## Version Alignment

- Spark cluster: 3.5.6 (`Dockerfile.spark` + `docker-compose.yml` build target)
- PySpark (driver): 3.5.6 (`requirements.txt`)
- Python: 3.11 on driver and executors (`Dockerfile.spark` compiles 3.11 and sets `PYSPARK_PYTHON`)

Tip: After recreating the Jupyter container, run `docker compose exec jupyter pip install -r work/requirements.txt` to restore Python packages.

## Data Upload Scripts (Cross‑platform)

- PowerShell (Windows): `scripts/load_to_hdfs.ps1 -DataDir "data/movielens/32m"`

## How The Recommender Works

- Data + Training (Spark)
  - Reads ratings.csv and movies.csv from the MovieLens dataset.
  - Trains an ALS model (matrix factorization) with PySpark ML.
  - Writes Parquet artifacts under `outputs/`:
    - `movies_meta` (movieId, title, genres, year)
    - `user_topn` (Top‑N recommendations per user, includes score and metadata)
    - `item_factors` (movieId, latent feature vector)
    - `popularity` (pop_score by movie, plus metadata)
  - For large datasets (32M), artifacts are written as multiple Parquet part files (no single huge file) for stability and speed.

- API (FastAPI)
  - `GET /recommendations/user/{user_id}` → personalized Top‑N (filters: genres/year range)
  - `GET /recommendations/item/{movie_id}` → similar items (cosine on normalized item factors)
  - `GET /popular` → popular titles (by genre or overall)
  - `GET /users` → valid user IDs from `user_topn` (sorted ascending)
  - `GET /years` → valid year range from `movies_meta`
  - `POST /feedback` → records list/favorite events to JSONL (optional; powers “Most Popular”)
  - `GET /feedback/summary` → decayed + weighted popularity from feedback, blended with baseline

- UI (at `/ui`)
  - Pick user (dataset‑backed dropdown with search‑in‑menu)
  - Pick genres (multi‑select chips) and year presets (chips)
  - Recommend renders immediately; posters load in the background (fast, non‑blocking)
  - Dedicated grid page for each genre tag

## Quick Start (Local)

1) Train + Write Artifacts (light settings for validation)

```
python -m src.recommendation \
  --master-local \
  --ratings-path data/movielens/32m/ratings.csv \
  --movies-path data/movielens/32m/movies.csv \
  --rank 8 --max-iter 4 --reg 0.1 \
  --topn-k 20 \
  --write-artifacts --precompute-dir outputs
```

2) Run API + UI

```
$env:PRECOMPUTE_DIR = (Resolve-Path .\outputs).Path
py -m uvicorn api.main:app --reload --port 8000
```

Open: http://127.0.0.1:8000/ui (Ctrl+F5 for a hard refresh)

## Notes

- Posters (optional): set `TMDB_API_KEY` if you want thumbnails. The UI caps lookups and timeouts to stay responsive.
- 404s like `/.well-known/appspecific/...` were silenced; `favicon.ico` is handled and linked.
- Bash (Linux/macOS/WSL): `bash scripts/load_to_hdfs.sh`

## Future Improvements

- Sweep ALS hyperparameters and support implicit feedback mode.
- Add popularity baseline and hybrid re‑ranking for cold‑start.
- Persist models and serve recommendations behind a lightweight API.
- Batch inference to precompute top‑N per user and materialize to Parquet.

## Test Status

- ✅ All tests pass under Python 3.11, Spark 3.3+, Hadoop 3.3.1, pyarrow 14+.
- Tested on Windows 10 with JDK 17.
