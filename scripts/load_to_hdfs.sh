#!/usr/bin/env bash
set -euo pipefail

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

run_hdfs() {
  local cmd=("hdfs" "dfs" "$@")
  if [[ "${USE_DOCKER}" == "true" ]]; then
    docker exec -i "${HDFS_CONTAINER}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

run_hdfs -mkdir -p "${HDFS_TARGET}"

for csv in "${csv_files[@]}"; do
  file_name=$(basename "${csv}")
  echo "Uploading ${csv} -> ${HDFS_URI}${HDFS_TARGET}/${file_name}" >&2
  run_hdfs -put -f "${csv}" "${HDFS_URI}${HDFS_TARGET}/${file_name}"
done

echo "Upload complete."
