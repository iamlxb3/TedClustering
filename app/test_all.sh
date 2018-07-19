#!/bin/bash


# MiniBatchKMeans
#min_n=5
#max_n=6
#features=( 'tfidf' 'tf' )
#for ((i=0;i<${#features[@]};++i));do
#    for ((n=min_n;n<=max_n;++n));do
#        python clustering.py -f ${features[i]} -c MiniBatchKMeans --n_cluster $n
#    done
#done
#

# MiniBatchKMeans + lsa
min_n=5
max_n=30
features=( 'tfidf' 'tf' )
lsa_ns=( 100 300 500 700 900 )
for ((j=0;j<${#lsa_ns[@]};++j));do
    for ((i=0;i<${#features[@]};++i));do
        for ((n=min_n;n<=max_n;++n));do
            python clustering.py -f ${features[i]} -c MiniBatchKMeans --n_cluster $n --preprocess lsa --lsa_n ${lsa_ns[j]}
        done
    done
done
#