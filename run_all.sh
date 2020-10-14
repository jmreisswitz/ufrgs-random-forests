#!/bin/bash

ntree_values=( 10 25 50 )
datasets=("house-votes-84.tsv" "wine-recognition.tsv")
for dataset in "${datasets[@]}"
do
  for ntree in "${ntree_values[@]}"
  do
    echo "$dataset with $ntree trees"
    python3 main.py --dataset $dataset --target_column target --ntree $ntree > "results/$dataset-$ntree.csv"
  done
done
