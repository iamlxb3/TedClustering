#!/bin/bash

n_cluster=( 5 6 7 8 9 10 11 12 13 14 15 )
for ((i=0;i<${#n_cluster[@]};++i));do
    n_cluster=${n_cluster[i]}
    python clustering.py -f tfidf -c MiniBatchKMeans --n_cluster $n_cluster
done