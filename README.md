# CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud

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
│       └── 25m/                      # Example extracted dataset variant
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

```
docker compose up -d
```

(If you are using an older Docker Compose release, the legacy
`docker-compose up -d` command works as well.) This command activates Hadoop
HDFS and Apache Spark services for data processing

3. Access Jupyter Notebook
   
If you are using Jupyter Notebook within Docker, you can access it at:

`
http://localhost:8888/
`

Otherwise, you can run the `.ipynb` file in any local or cloud-based Jupyter environment.

### Docker image troubleshooting

If Docker reports an error similar to
`failed to resolve reference "docker.io/bitnami/spark:3.5.0"`, it means your
local clone is still using an older Compose file that points at the retired
`3.5.0` tag. First, make sure you are on the latest repo revision (for example,
`git pull origin main`). Then pull the current Bitnami Spark tag manually and
re-run Compose:

```
docker pull bitnami/spark:3.5
docker compose up -d
```

The Compose file now references `bitnami/spark:3.5`, which tracks the latest
Spark 3.5 patch release published by Bitnami.

4. Confirm your local clone is connected to GitHub

If you are working across multiple Windows profiles and want to verify where your
changes live, list the configured remotes:

```
git remote -v
```

If no remotes are returned, the work is only stored locally. You can also run
the helper script `python scripts/check_git_sync.py` for a quick summary of the
current branch and remote status.

When you need to connect the clone to your GitHub fork (replace the URL with
your repository if different) and push the branch that contains your latest
commits, run:

```
git remote add origin https://github.com/TsungTseTu122/CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud.git
git push -u origin work
```

After pushing, confirm the branch or pull request on GitHub to ensure the
changes are available remotely.

### Opening a pull request on GitHub

Pushing your branch uploads the commits, but a pull request (PR) is what makes
those commits visible for review and merging. To open a PR:

1. Visit your fork on GitHub (for example,
   `https://github.com/TsungTseTu122/CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud`).
2. If GitHub detects the recently pushed branch, click **Compare & pull request**.
   Otherwise, choose **Pull requests → New pull request** and select your branch
   (e.g., `work`) as the source and `main` as the target.
3. Fill in a descriptive title and summary of the changes, then click
   **Create pull request**.

If you do not see the branch listed, double-check that the push succeeded and
that you pushed to the expected remote URL.

### Publishing follow-up commits

Whenever you make additional updates (such as refining the README or
adjusting `docker-compose.yml`), commit them locally and push again:

```
git add .
git commit -m "Describe your change"
git push
```

You can repeat these commands as often as needed; each push updates the
branch on GitHub with your latest work.

## Dataset

### 1. Download MovieLens automatically

Run the helper script to download and extract the dataset variant you need (25M by default):

```
python scripts/download_movielens.py --variant 25m
```

The command stores the CSV files under `data/movielens/<variant>/` and records a checksum so repeat runs skip the download unless you add `--force`.

### 2. Load data into HDFS

After the CSVs are available locally, push them into HDFS so Spark can read them inside the cluster:

```
bash scripts/load_to_hdfs.sh
```

The script uploads every CSV in `data/movielens/25m` to `hdfs://namenode:8020/user/hadoop/movielens/`. Override the source directory or container name with the `DATA_DIR`, `HDFS_TARGET`, and `HDFS_CONTAINER` environment variables if your setup differs (for example on Windows with WSL you can run `wsl bash scripts/load_to_hdfs.sh`).

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

## Future Improvements

- Enhance model performance with deep learning-based collaborative filtering.

- Optimize Spark configurations for faster data processing.

- Implement real-time recommendations using streaming data analysis.
