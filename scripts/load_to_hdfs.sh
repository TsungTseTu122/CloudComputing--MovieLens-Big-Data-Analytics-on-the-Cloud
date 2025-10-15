#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/load_to_hdfs.sh [DATA_DIR]
# Env:
#   HDFS_TARGET=/user/hadoop/movielens
#   HDFS_URI=hdfs://namenode:8020
#   HDFS_CONTAINER=namenode
#   USE_DOCKER=true|false (if true, copies files into the container before put)

DATA_DIR=${1:-data/movielens/25m}
HDFS_TARGET=${HDFS_TARGET:-/user/hadoop/movielens}
HDFS_URI=${HDFS_URI:-hdfs://namenode:8020}
HDFS_CONTAINER=${HDFS_CONTAINER:-namenode}
USE_DOCKER=${USE_DOCKER:-true}

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Data directory ${DATA_DIR} not found" >&2
  exit 1
fi

mapfile -t csv_files < <(find "${DATA_DIR}" -maxdepth 1 -type f -name '*.csv')
if [[ ${#csv_files[@]} -eq 0 ]]; then
  echo "No CSV files located in ${DATA_DIR}" >&2
  exit 1
fi

if [[ "${USE_DOCKER}" == "true" ]]; then
  # Prepare container temp dir and HDFS target
  docker exec -i "${HDFS_CONTAINER}" bash -lc "mkdir -p /tmp/movielens"
  docker exec -i "${HDFS_CONTAINER}" hdfs dfs -mkdir -p "${HDFS_TARGET}"

  for csv in "${csv_files[@]}"; do
    file_name=$(basename "${csv}")
    echo "Uploading ${csv} -> ${HDFS_URI}${HDFS_TARGET}/${file_name}" >&2
    docker cp "${csv}" "${HDFS_CONTAINER}:/tmp/movielens/${file_name}"
    docker exec -i "${HDFS_CONTAINER}" hdfs dfs -put -f \
      "/tmp/movielens/${file_name}" \
      "${HDFS_URI}${HDFS_TARGET}/${file_name}"
  done
else
  # Direct host CLI (assumes HDFS CLI installed locally)
  hdfs dfs -mkdir -p "${HDFS_TARGET}"
  for csv in "${csv_files[@]}"; do
    file_name=$(basename "${csv}")
    echo "Uploading ${csv} -> ${HDFS_URI}${HDFS_TARGET}/${file_name}" >&2
    hdfs dfs -put -f "${csv}" "${HDFS_URI}${HDFS_TARGET}/${file_name}"
  done
fi

echo "Upload complete."
