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
The dataset is not included in this repository due to its size. You can download it from:

MovieLens 32M Dataset: https://grouplens.org/datasets/movielens/

After downloading, place the data files inside the data/ directory.

## Key Features

- Scalable Data Processing: Uses Spark on a cloud-based infrastructure for efficient handling of large-scale datasets.

- Movie Recommendation Models: Implements clustering and classification techniques to analyze user preferences.

- Association Rule Mining: Identifies patterns in user interactions to improve recommendation quality.

- Containerized Deployment: Dockerized services for easy setup and replication.

## Future Improvements

- Enhance model performance with deep learning-based collaborative filtering.

- Optimize Spark configurations for faster data processing.

- Implement real-time recommendations using streaming data analysis.
