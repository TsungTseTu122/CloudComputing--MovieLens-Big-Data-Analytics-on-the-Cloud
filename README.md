# CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud

## Overview
This project analyzes the MovieLens dataset using PySpark, Hadoop HDFS, and Docker to perform clustering, classification, and association rule mining on user-movie interactions. The system runs in a containerized cloud environment with Spark clusters, enabling scalable big data processing.

## Project Repository
CloudComputing--MovieLens-Big-Data-Analytics-on-the-Cloud
```
│── README.md                  # Project Overview & Usage Guide
│── requirements.txt            # Python dependencies 
│── docker-compose.yml          # Docker setup for Spark & Hadoop
│── notebooks/
│   ├── Project.ipynb           # Jupyter Notebook with data analysis
│── reports/
│   ├── TsungTse_Tu_s4780187_proposal.docx  # Project proposal
│── data/                       # Placeholder for datasets (need download)
│   ├── MovieLens_Dataset_Info.txt             # Dataset license and source information
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
