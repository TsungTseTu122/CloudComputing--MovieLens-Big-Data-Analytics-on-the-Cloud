# MovieLens Big Data Analytics on the Cloud

## Overview

This project demonstrates end-to-end big data analytics on the MovieLens dataset using Apache Spark, Hadoop HDFS, and Docker. It includes two distinct analysis approaches:

1. **Recommender System (Recommender.ipynb)**: Production-ready collaborative filtering using ALS (Alternating Least Squares) to generate personalized movie recommendations
2. **Exploratory Data Analysis (EDA.ipynb)**: Statistical analysis, visualization, and machine learning for understanding rating patterns and movie characteristics

The system is designed to scale from local development to distributed cloud deployment, with precomputed outputs served via a FastAPI web service.

---

## Project Structure

```
CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud/
|
|-- README.md                         # This file
|-- requirements.txt                  # Python dependencies (PySpark, FastAPI, etc.)
|-- docker-compose.yml                # Docker services (Hadoop, Spark, Jupyter)
|-- Dockerfile.spark                  # Custom Spark image with Python 3.11
|
|-- data/                             # MovieLens datasets (auto-downloaded)
|   |-- MovieLens_Dataset_Info.txt    # License and source information
|   |-- movielens/
|       |-- 32m/                      # MovieLens 32M dataset
|       |   |-- ratings.csv           # User ratings (userId, movieId, rating, timestamp)
|       |   |-- movies.csv            # Movie metadata (movieId, title, genres)
|       |   |-- tags.csv              # User-generated tags
|       |   |-- links.csv             # Links to IMDB and TMDB
|       |-- latest-small/             # Small dataset for testing
|
|-- notebooks/
|   |-- Recommender.ipynb             # ALS recommender system (train, evaluate, export)
|   |-- EDA.ipynb                     # Exploratory analysis and ML profiling
|
|-- scripts/
|   |-- download_movielens.py         # Download and validate MovieLens datasets
|   |-- load_to_hdfs.sh               # Upload CSVs to HDFS (Linux/WSL)
|   |-- load_to_hdfs.ps1              # Upload CSVs to HDFS (Windows PowerShell)
|
|-- src/
|   |-- __init__.py
|   |-- recommendation.py             # ALS training and inference utilities
|
|-- api/
|   |-- main.py                       # FastAPI service for serving recommendations
|
|-- web/
|   |-- index.html                    # Netflix-style web UI
|   |-- app.js                        # Frontend JavaScript
|   |-- styles.css                    # UI styling
|
|-- outputs/                          # Precomputed artifacts (created by notebooks)
|   |-- user_topn/                    # Top-N recommendations per user (Parquet)
|   |-- popularity/                   # Popular movies for cold-start (Parquet)
|   |-- movies_meta/                  # Movie metadata (Parquet)
|   |-- item_factors/                 # ALS item latent factors (Parquet)
|
|-- tests/
|   |-- test_recommendation.py        # Unit tests for ALS pipeline
|   |-- test_api.py                   # API endpoint tests
|
|-- conf/                             # Hadoop configuration files
|-- reports/                          # Project proposal and documentation
```

---

## Architecture

### Data Flow

```
[MovieLens CSVs] 
    |
    v
[Local Storage / HDFS]
    |
    v
[Spark ALS Training] --> [Recommender.ipynb]
    |
    v
[Precomputed Artifacts] --> outputs/ (Parquet)
    |
    v
[FastAPI Service] --> api/main.py
    |
    v
[Web UI] --> web/index.html
```

### Components

1. **Data Storage**
   - Local filesystem: `data/movielens/32m/` (for development)
   - HDFS: `hdfs://namenode:8020/user/hadoop/movielens/` (for cluster mode)

2. **Training (Spark)**
   - Notebook: `Recommender.ipynb` (interactive development)
   - CLI: `python -m src.recommendation` (batch processing)
   - Algorithm: ALS (Alternating Least Squares) collaborative filtering
   - Output: Parquet files in `outputs/` directory

3. **Serving (FastAPI)**
   - Load precomputed recommendations from Parquet
   - REST API endpoints for user/item recommendations
   - Cold-start handling with popularity baseline
   - Optional TMDB integration for movie posters

4. **Web UI**
   - Netflix-style interface at `/ui`
   - User selection, genre filtering, year range
   - Real-time poster loading from TMDB API

---

## Two Notebooks Explained

### 1. Recommender.ipynb - Collaborative Filtering Recommender System

**Purpose**: Train and evaluate a production-ready recommendation model using Spark ALS.

**What it does**:
- Loads MovieLens ratings and movies from CSV (local or HDFS)
- Cleans and filters data (removes sparse users/movies)
- Splits data: 80% train, 10% validation, 10% test
- Trains ALS model with hyperparameter tuning (rank, regParam)
- Evaluates using:
  - RMSE (Root Mean Squared Error) for prediction accuracy
  - Precision@K and Recall@K for ranking quality
- Generates top-N recommendations for all users
- Handles cold-start users with popularity baseline
- Exports outputs to Parquet for API serving:
  - `user_topn/`: Top-100 recommendations per user with metadata
  - `popularity/`: Popular movies ranked by rating count * avg rating
  - `movies_meta/`: Movie metadata (title, genres, year)
  - `item_factors/`: ALS latent factors for similarity computation

**Key Features**:
- Self-contained: runs without `src/` module (though compatible)
- Configurable data paths (local or HDFS)
- Explicit hyperparameter search over validation set
- Clear evaluation metrics with interpretation
- Cold-start fallback logic
- Ready for production serving

**Run Time**: ~10-30 minutes depending on dataset size and hardware.

---

### 2. EDA.ipynb - Exploratory Data Analysis

**Purpose**: Explore rating patterns, movie characteristics, and temporal trends. Perform ML profiling (NOT recommendations).

**What it does**:
- Loads and preprocesses MovieLens data (ratings, movies, tags)
- Exploratory statistics:
  - Rating distribution and outliers
  - Movie release year histogram
  - Genre popularity and trends over time
  - Tag word clouds
  - Average rating per genre and year
- Machine learning for movie profiling:
  - K-Means clustering: group movies by year and rating
  - Random Forest classification: predict rating class (Low/Medium/High)
  - Random Forest regression: predict average movie rating
  - Association rule mining: discover genre co-occurrence patterns
- Visualizations using Plotly (interactive charts)

**What it does NOT do**:
- **Does NOT generate personalized user recommendations**
- **Does NOT train a collaborative filtering model**
- This is for data exploration and feature engineering only

**Run Time**: ~15-45 minutes depending on dataset size.

---

## Setup and Installation

### Prerequisites

- **Docker**: Version 20.10 or later
- **Python**: 3.11 (for local development)
- **Java**: JDK 11 or 17 (for Spark)
- **Git**: For cloning the repository

### Quick Start (Local Mode)

1. **Clone the repository**

```bash
git clone https://github.com/TsungTseTu122/CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud.git
cd CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud
```

2. **Download MovieLens data**

```bash
python scripts/download_movielens.py --variant 32m
```

This downloads and extracts the 32M dataset to `data/movielens/32m/`. For testing, use `--variant latest-small`.

3. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

4. **Run Recommender notebook**

```bash
jupyter notebook notebooks/Recommender.ipynb
```

- Set `DATA_DIR = "data/movielens/32m"` in the configuration cell
- Run all cells sequentially
- Outputs will be saved to `outputs/` directory

5. **Start the API server**

```bash
# Set output directory (PowerShell)
$env:PRECOMPUTE_DIR = (Resolve-Path .\outputs).Path

# Or Bash
export PRECOMPUTE_DIR=$(pwd)/outputs

# Start FastAPI
python -m uvicorn api.main:app --reload --port 8000
```

6. **Access the Web UI**

Open http://127.0.0.1:8000/ui in your browser.

---

## Cluster Mode (Docker)

### Start Docker Services

```bash
docker-compose up -d
```

This starts:
- Hadoop NameNode (http://localhost:9870)
- Hadoop DataNode (http://localhost:9864)
- Spark Master (http://localhost:8080)
- Spark Worker
- Jupyter Notebook (http://localhost:8888)

### Load Data to HDFS

**Linux/macOS/WSL**:

```bash
bash scripts/load_to_hdfs.sh
```

**Windows PowerShell**:

```powershell
.\scripts\load_to_hdfs.ps1 -DataDir "data\movielens\32m"
```

This uploads CSVs to `hdfs://namenode:8020/user/hadoop/movielens/`.

### Run Recommender on Cluster

```bash
docker compose exec \
  -e SPARK_HOME=/home/jovyan/spark-3.5.6-bin-hadoop3 \
  -w /home/jovyan/work \
  jupyter \
  python -m src.recommendation \
    --master spark://spark-master:7077 \
    --ratings-path hdfs://namenode:8020/user/hadoop/movielens/ratings.csv \
    --movies-path  hdfs://namenode:8020/user/hadoop/movielens/movies.csv \
    --rank 10 --max-iter 5 --reg 0.1 \
    --write-artifacts --precompute-dir outputs --topn-k 100
```

### Access Notebooks in Cluster

1. Open http://localhost:8888
2. Navigate to `notebooks/Recommender.ipynb`
3. Update data paths to HDFS:

```python
DATA_DIR = "hdfs://namenode:8020/user/hadoop/movielens"
```

4. Change Spark master to cluster mode:

```python
spark = SparkSession.builder.master("spark://spark-master:7077")...
```

---

## Outputs and Artifacts

The recommender system exports precomputed artifacts to `outputs/` for efficient serving.

### Output Directory Structure

```
outputs/
|-- user_topn/                # Top-N recommendations per user
|   |-- part-00000.parquet
|   |-- ...
|   |-- _SUCCESS
|
|-- popularity/               # Popular movies (cold-start fallback)
|   |-- part-00000.parquet
|   |-- _SUCCESS
|
|-- movies_meta/              # Movie metadata
|   |-- part-00000.parquet
|   |-- _SUCCESS
|
|-- item_factors/             # ALS item latent factors
    |-- part-00000.parquet
    |-- _SUCCESS
```

### Artifact Schemas

**user_topn** (User Recommendations):

```
userId: int
movieId: int
rank: int
predicted_rating: float
clean_title: string
full_title: string
genres: string
year: string
```

**popularity** (Popular Movies):

```
movieId: int
clean_title: string
genres: string
year: string
rating_count: long
avg_rating: double
popularity_score: double  # rating_count * avg_rating
```

**movies_meta** (Movie Metadata):

```
movieId: int
clean_title: string
title: string
genres: string
year: string
```

**item_factors** (ALS Factors):

```
id: int  # movieId
features: array<float>  # latent factor vector
```

### How the API Uses Artifacts

1. **User Recommendations**: `/recommendations/user/{userId}`
   - Reads `user_topn/` Parquet
   - Filters by userId, optional genre/year range
   - Returns top-N sorted by predicted_rating

2. **Item Similarity**: `/recommendations/item/{movieId}`
   - Reads `item_factors/` Parquet
   - Computes cosine similarity between item vectors
   - Returns top-N similar movies

3. **Popular Movies**: `/popular`
   - Reads `popularity/` Parquet
   - Filters by genre if specified
   - Returns top-N sorted by popularity_score

4. **Cold Start**: Fallback for new users
   - If userId not in `user_topn/`, return `/popular` results

---

## Evaluation Metrics

### RMSE (Root Mean Squared Error)

- **Definition**: Square root of average squared prediction error
- **Formula**: `sqrt(mean((prediction - actual)^2))`
- **Interpretation**: Lower is better; measures prediction accuracy in rating units
- **Typical Values**: 0.8-1.2 for MovieLens (on 1-5 scale)

### Precision@K

- **Definition**: Fraction of recommended items that are relevant
- **Formula**: `(# relevant items in top-K) / K`
- **Interpretation**: Higher is better; measures recommendation accuracy
- **Example**: If 7 out of 10 recommended movies are relevant (rated >= 4.0), Precision@10 = 0.7

### Recall@K

- **Definition**: Fraction of relevant items that are recommended
- **Formula**: `(# relevant items in top-K) / (total # relevant items)`
- **Interpretation**: Higher is better; measures recommendation coverage
- **Example**: If user has 20 relevant movies total and 7 are in top-10, Recall@10 = 0.35

### Trade-offs

- **Precision vs. Recall**: Higher K increases recall but may decrease precision
- **RMSE vs. Ranking Metrics**: RMSE measures prediction accuracy; P@K and R@K measure ranking quality
- **Use Case**: For top-N recommendations, optimize P@K and R@K rather than RMSE

---

## Cold Start Handling

### Problem

- **User Cold Start**: New users with no rating history
- **Item Cold Start**: New movies with no ratings

### Solution: Popularity Baseline

1. **Compute Popularity Score**:
   - `popularity_score = rating_count * avg_rating`
   - Filter movies with `rating_count >= 50` (configurable threshold)

2. **Fallback Logic**:
   ```python
   if user_id in trained_users:
       return ALS_recommendations(user_id)
   else:
       return popular_movies(top_n=10)
   ```

3. **Optional Enhancements**:
   - Genre-specific popularity (e.g., popular Horror movies)
   - Recency weighting (favor recent releases)
   - Hybrid: combine popularity with content-based features

### Limitations

- Popularity baseline ignores individual preferences
- Does not solve item cold start (new movies not recommended until rated)
- Future work: hybrid models combining collaborative and content-based filtering

---

## Limitations and Future Work

### Current Limitations

1. **Cold Start**:
   - New users receive only popular movies (no personalization)
   - New movies cannot be recommended until rated

2. **Sparsity**:
   - Most users rate < 50 movies (very sparse interaction matrix)
   - ALS struggles with users who have few ratings

3. **Popularity Bias**:
   - Model may favor popular items over niche content
   - Long-tail movies are under-recommended

4. **No Content Features**:
   - ALS uses only user-item interactions
   - Ignores movie metadata (genres, tags, year)

5. **Static Model**:
   - Requires periodic retraining as new ratings arrive
   - No online learning or incremental updates

6. **Scalability**:
   - Large datasets (100M+ ratings) require cluster resources
   - Memory and shuffle optimization needed

7. **Evaluation**:
   - Offline metrics (RMSE, P@K, R@K) may not reflect online performance
   - No A/B testing or engagement tracking

### Future Improvements

1. **Hybrid Models**:
   - Combine collaborative filtering with content-based features
   - Use genre, year, tags, cast, director for cold-start

2. **Implicit Feedback**:
   - Model view/click/watch events in addition to ratings
   - Use ALS with `implicitPrefs=True`

3. **Real-time Updates**:
   - Implement online learning (e.g., FTRL, SGD)
   - Update recommendations as users rate new movies

4. **Diversity and Serendipity**:
   - Add diversity constraints (avoid recommending similar items)
   - Boost serendipitous recommendations (unexpected but relevant)

5. **Explainability**:
   - Provide reasons for recommendations (e.g., "Because you liked Inception")
   - Show item-to-item paths in the interaction graph

6. **Production Serving**:
   - Deploy model to production with low-latency serving (Redis, TensorFlow Serving)
   - Implement caching and precomputation strategies

7. **A/B Testing**:
   - Evaluate recommendations with user engagement metrics (CTR, watch time)
   - Compare multiple algorithms in production

8. **Advanced Metrics**:
   - NDCG (Normalized Discounted Cumulative Gain)
   - MAP (Mean Average Precision)
   - Coverage and novelty metrics

---

## Dataset Information

### MovieLens 32M

- **Source**: https://grouplens.org/datasets/movielens/
- **Size**: 32 million ratings, 62,000 movies, 162,000 users
- **Files**:
  - `ratings.csv`: userId, movieId, rating (0.5-5.0), timestamp
  - `movies.csv`: movieId, title, genres (pipe-separated)
  - `tags.csv`: userId, movieId, tag, timestamp
  - `links.csv`: movieId, imdbId, tmdbId

- **License**: Creative Commons (CC BY-NC 4.0)
- **Citation**: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1-19:19.

### MovieLens Latest-Small

- **Size**: 100,000 ratings, 9,000 movies, 600 users
- **Use Case**: Testing and development
- **Download**: `python scripts/download_movielens.py --variant latest-small`

---

## Quick Reference

### Common Commands

```bash
# Download data
python scripts/download_movielens.py --variant 32m

# Start Docker cluster
docker-compose up -d

# Load data to HDFS
bash scripts/load_to_hdfs.sh

# Train model (local)
jupyter notebook notebooks/Recommender.ipynb

# Start API
$env:PRECOMPUTE_DIR = (Resolve-Path .\outputs).Path
python -m uvicorn api.main:app --reload --port 8000

# Open Web UI
http://localhost:8000/ui

# Run tests
pytest tests/ -v
```

### Key URLs

- Jupyter: http://localhost:8888
- Spark Master UI: http://localhost:8080
- HDFS NameNode: http://localhost:9870
- API Swagger: http://localhost:8000/docs
- Web UI: http://localhost:8000/ui

---

## License

This project is for educational purposes. The MovieLens dataset is licensed under Creative Commons (CC BY-NC 4.0).

---

## Acknowledgments

- GroupLens Research for the MovieLens dataset
- Apache Spark and Hadoop communities
- FastAPI and Plotly open-source projects
