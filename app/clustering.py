import sys
import os
import pandas as pd
import argparse
import numpy as np
import shutil
import ipdb

sys.path.append('..')
from funcs.clusters import clusters
from funcs.helpers import clustering_eval
from funcs.helpers import metric_record
from funcs.helpers import wordcloud_analysis
from sklearn.decomposition import TruncatedSVD
"""
python clustering.py --clear -f tf -c KMeans
python clustering.py --clear -f tf -c MiniBatchKMeans --n_cluster 5
python clustering.py --clear -f tf -c Spectral
python clustering.py --clear -f tf -c Agglomerative --linkage ward
python clustering.py --clear -f tf -c DBSCAN --eps 0.5 --min_samples 5
"""


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', dest='clear', help='flag to clear the previous output',
                        default=False, action='store_true')
    parser.add_argument('--cloud', dest='is_wordcloud', help='whether to generate wordcloud',
                        default=False, action='store_true')
    parser.add_argument('-f', '--feature', dest='feature', help='valid features for clustering: tf, tfidf',
                        required=True, type=str, choices=('tf, tfidf'))
    parser.add_argument('-c', '--cluster', dest='cluster', help='type valid clusters for clustering',
                        required=True, type=str, choices=('KMeans',
                                                          'MiniBatchKMeans',
                                                          'Spectral',
                                                          'Agglomerative',
                                                          'DBSCAN'
                                                          ))
    parser.add_argument('--n_cluster', dest='n_cluster', help='n_cluster for KMeans and MiniBatchKMeans',
                        type=int)
    parser.add_argument('--linkage', dest='linkage', help='linkage for agglomerative clustering',
                        type=str, choices=('ward', 'complete', 'average'))
    parser.add_argument('--eps', dest='eps', help='eps for DBSCAN',
                        type=float)
    parser.add_argument('--min_samples', dest='min_samples', help='eps for DBSCAN',
                        type=int)
    parser.add_argument('--lsa_n', dest='lsa_n', help='n_components for latent semantic analysis',
                        type=int)
    parser.add_argument('--preprocess', dest='preprocess', help='data_preprocess',
                        type=str)
    args = parser.parse_args()

    if args.cluster == 'KMeans' or args.cluster == 'MiniBatchKMeans' and not args.n_cluster:
        parser.error('KMeans and MiniBatchKMeans require n_cluster')

    if args.cluster == 'Agglomerative' and not args.linkage:
        parser.error('Agglomerative requires linkage')

    if args.cluster == 'DBSCAN' and not args.eps:
        parser.error('DBSCAN requires eps')

    if args.cluster == 'DBSCAN' and not args.min_samples:
        parser.error('DBSCAN requires min_samples')

    if args.preprocess == 'las' and not args.lsa_n:
        parser.error('preprocess requires lsa_n')


    cluster = args.cluster
    feature = args.feature
    clear = args.clear
    is_wordcloud = args.is_wordcloud
    linkage = args.linkage
    eps = args.eps
    min_samples = args.min_samples
    n_cluster = args.n_cluster
    lsa_n = args.lsa_n
    preprocess = args.preprocess

    return feature, cluster, clear, linkage, eps, min_samples, is_wordcloud, n_cluster, lsa_n, preprocess


def main():
    # set path
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data', 'TED')
    ted_path = os.path.join(data_dir, 'ted.csv')
    output_dir = os.path.join(top_dir, 'output')
    results_csv_path = os.path.join(top_dir, 'output', 'metrics.csv')

    # argument parse
    feature, cluster, clear, linkage, eps, min_samples, is_wordcloud, n_clusters, lsa_n, preprocess = args_parse()

    # clear the output
    if clear:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print("Remove all content in {}".format(output_dir))

    # load the ted df
    ted_df = pd.read_csv(ted_path, encoding='utf-8')
    print("load ted df done!")

    # read the tf matrix, idf arr
    tf_matrix_path = os.path.join(data_dir, "tf.npy")
    idf_arr_path = os.path.join(data_dir, "idf.npy")
    tf_matrix, words = np.load(tf_matrix_path)
    idf_arr = np.load(idf_arr_path)
    print("load tf_matrix and idf array done! Tf matrix shape: {}, idf arrar shape: {}"
          .format(tf_matrix.shape, idf_arr.shape))

    # read M x N feature matrix (M -> number of documents, N -> number of features)
    matrix_path = os.path.join(data_dir, "{}.npy".format(feature))

    matrix, words = np.load(matrix_path)
    print("Load document-feature matrix done! Shape: {}, feature: {}".format(matrix.shape, feature))

    # do clustering
    X = matrix[0:20]  # TODO
    #

    # pre process data & feature
    if preprocess == 'lsa':
        lsa = TruncatedSVD(n_components=lsa_n)
        if lsa_n > min(X.shape):
            print("Warning, lsa n component won't be greated than max rank!") # TODO, maybe?
        X = lsa.fit_transform(X)
        print("Lsa done! Shape: {}".format(X.shape))
    #

    parameter_dict = {
        'KMeans': (X, n_clusters),
        'MiniBatchKMeans': (X, n_clusters),
        'Spectral': (X, n_clusters),
        'Agglomerative': (X, n_clusters, linkage),
        'DBSCAN': (X, n_clusters, eps, min_samples),
    }
    labels = clusters[cluster](*parameter_dict[cluster])

    # intrinsic evaluation
    sil_val = clustering_eval(labels, X, mode='sil')
    cal_val = clustering_eval(labels, X, mode='cal')

    # record metric
    metric_record(results_csv_path, cluster, sil_val, cal_val,
                  n_clusters=n_clusters,
                  eps=eps,
                  linkage=linkage,
                  min_samples=min_samples,
                  feature=feature,
                  preprocess=preprocess,
                  )

    # analysis word distribution and word cloud
    if is_wordcloud:
        wordcloud_analysis(cluster, output_dir, labels, tf_matrix, idf_arr, words, ted_df)


if __name__ == '__main__':
    main()
