import sys
import os
import pandas as pd
import argparse
import numpy as np
import shutil

sys.path.append('..')
from funcs.clusters import clusters
from funcs.helpers import clustering_eval
from funcs.helpers import metric_record
from funcs.helpers import wordcloud_analysis

"""
python clustering.py --clear -f tf -c KMeans
python clustering.py --clear -f tf -c MiniBatchKMeans
python clustering.py --clear -f tf -c Spectral
python clustering.py --clear -f tf -c Agglomerative --linkage ward
python clustering.py --clear -f tf -c DBSCAN --eps 0.5 --min_samples 5
"""


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', dest='clear', help='flag to clear the previous output',
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
    parser.add_argument('--linkage', dest='linkage', help='linkage for agglomerative clustering',
                        type=str, choices=('ward', 'complete', 'average'))
    parser.add_argument('--eps', dest='eps', help='eps for DBSCAN',
                        type=float)
    parser.add_argument('--min_samples', dest='min_samples', help='eps for DBSCAN',
                        type=int)

    args = parser.parse_args()

    if args.cluster == 'Agglomerative' and not args.linkage:
        parser.error('Agglomerative requires linkage')

    if args.cluster == 'DBSCAN' and not args.eps:
        parser.error('DBSCAN requires eps')

    if args.cluster == 'DBSCAN' and not args.min_samples:
        parser.error('DBSCAN requires min_samples')

    cluster = args.cluster
    feature = args.feature
    clear = args.clear
    linkage = args.linkage
    eps = args.eps
    min_samples = args.min_samples
    return feature, cluster, clear, linkage, eps, min_samples


def main():
    # set path
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data', 'TED')
    ted_path = os.path.join(data_dir, 'ted.csv')
    output_dir = os.path.join(top_dir, 'output')
    results_csv_path = os.path.join(top_dir, 'output', 'metrics.csv')

    # argument parse
    feature, cluster, clear, linkage, eps, min_samples = args_parse()

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
    n_clusters = 4
    X = matrix[0:20]  # TODO
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
    metric_record(results_csv_path, cluster, sil_val, cal_val, n_clusters=n_clusters)

    # analysis word distribution and word cloud
    wordcloud_analysis(cluster, output_dir, labels, tf_matrix, idf_arr, words, ted_df)


if __name__ == '__main__':
    main()
