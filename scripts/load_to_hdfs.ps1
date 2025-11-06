Param(
  [string]$DataDir = "data/movielens/32m",
  [string]$HdfsTarget = "/user/hadoop/movielens",
  [string]$HdfsUri = "hdfs://namenode:8020",
  [string]$HdfsContainer = "namenode"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
  throw "Data directory not found: $DataDir"
}

$csvFiles = Get-ChildItem -LiteralPath $DataDir -Filter *.csv -File | Select-Object -ExpandProperty FullName
if (-not $csvFiles -or $csvFiles.Count -eq 0) {
  throw "No CSV files found in $DataDir"
}

# Ensure tmp dir in container and HDFS target exist
docker exec -i $HdfsContainer bash -lc "mkdir -p /tmp/movielens"
docker exec -i $HdfsContainer hdfs dfs -mkdir -p $HdfsTarget

foreach ($csv in $csvFiles) {
  $fileName = Split-Path -Leaf $csv
  Write-Host "Uploading $csv -> $HdfsUri$HdfsTarget/$fileName"
  docker cp $csv "$HdfsContainer`:/tmp/movielens/$fileName"
  docker exec -i $HdfsContainer hdfs dfs -put -f "/tmp/movielens/$fileName" "$HdfsUri$HdfsTarget/$fileName"
}

Write-Host "Upload complete."

