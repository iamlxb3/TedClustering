import argparse
import os
import shutil
import sys
import ipdb
import numpy as np
import pandas as pd
import collections

sys.path.append('..')
from funcs.clusters import clusters
from funcs.helpers import clustering_in_eval
from funcs.helpers import metric_record
from funcs.helpers import wordcloud_analysis
from funcs.helpers import tsne_plot
from funcs.helpers import clustering_out_eval
from funcs.helpers import out_eval_prep
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import distance_matrix

"""
python clustering.py -f tfidf -c KMeans --n_cluster 10
python clustering.py -f tf -c MiniBatchKMeans --n_cluster 5 --preprocess lsa --lsa_n 100 --tsne
python clustering.py --clear -f tf -c Spectral
python clustering.py -f tfidf -c Agglomerative --linkage ward --n_cluster 10
python clustering.py -f tfidf -c Agglomerative --linkage complete --n_cluster 10
python clustering.py -f tfidf -c Agglomerative --linkage average --n_cluster 10

python clustering.py --clear -f tf -c DBSCAN --eps 0.5 --min_samples 5
"""


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', dest='clear', help='flag to clear the previous output',
                        default=False, action='store_true')
    parser.add_argument('--cloud', dest='is_wordcloud', help='whether to generate wordcloud',
                        default=False, action='store_true')
    parser.add_argument('-f', '--feature', dest='feature', help='valid features for clustering: tf, tfidf',
                        required=True, type=str, choices=('tf', 'tfidf'))
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
                        type=str, choices=('lsa',))
    parser.add_argument('--tsne', dest='tsne', help='tsne', default=False, action='store_true')
    parser.add_argument('--debug', action='store_true', help='debug_mode', default=False)

    args = parser.parse_args()
    if (args.cluster == 'KMeans' or args.cluster == 'MiniBatchKMeans' or args.cluster == 'Agglomerative')\
            and (not args.n_cluster):
        parser.error('KMeans and MiniBatchKMeans require n_cluster')

    if args.cluster == 'Agglomerative' and not args.linkage:
        parser.error('Agglomerative requires linkage')

    if args.cluster == 'DBSCAN' and not args.eps:
        parser.error('DBSCAN requires eps')

    if args.cluster == 'DBSCAN' and args.n_cluster:
        parser.error('DBSCAN should not set n_cluster!')

    if args.cluster == 'DBSCAN' and not args.min_samples:
        parser.error('DBSCAN requires min_samples')

    if args.preprocess == 'las' and not args.lsa_n:
        parser.error('preprocess requires lsa_n')

    if args.lsa_n == 'las' and not args.preprocess:
        parser.error('lsa_n requires preprocess')

    if args.tsne == True and not args.lsa_n:
        parser.error('tsne requires lsa_n')

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
    tsne = args.tsne
    debug = args.debug

    return feature, cluster, clear, linkage, eps, min_samples, is_wordcloud, n_cluster, lsa_n, preprocess, tsne, debug


def main():
    # set path
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data', 'TED')
    ted_path = os.path.join(data_dir, 'ted.csv')
    output_dir = os.path.join(top_dir, 'output')
    results_csv_path = os.path.join(top_dir, 'output', 'metrics.csv')

    # argument parse
    feature, cluster, clear, linkage, eps, min_samples,\
    is_wordcloud, n_clusters, lsa_n, preprocess, tsne, debug = args_parse()

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
    # X = matrix[0:20]  # TODO
    # fit_X = matrix  # TODO
    if debug:
        fit_X = matrix[0:20]  # TODO
        print("Debug mode! Data has been reduced!")
    else:
        fit_X = matrix
    #

    # pre process data & feature
    if preprocess == 'lsa':
        lsa = TruncatedSVD(n_components=lsa_n)
        if lsa_n > min(matrix.shape):
            print("Warning, lsa n component won't be greated than max rank!")  # TODO, maybe?
        fit_X = lsa.fit_transform(matrix)
        print("Lsa done! Shape: {}".format(fit_X.shape))
    #

    # compute distance metrics
    if cluster == 'DBSCAN':
        fit_X = distance_matrix(fit_X, fit_X)
        print("Distance metrics calculated!")

    parameter_dict = {
        'KMeans': (fit_X, n_clusters),
        'MiniBatchKMeans': (fit_X, n_clusters),
        'Spectral': (fit_X, n_clusters),
        'Agglomerative': (fit_X, n_clusters, linkage),
        'DBSCAN': (fit_X, n_clusters, eps, min_samples),
    }
    labels = clusters[cluster](*parameter_dict[cluster])
    if cluster == 'DBSCAN':
        n_clusters = len([x for x in collections.Counter(labels).keys() if x != -1])

    # intrinsic evaluation
    sil_val = clustering_in_eval(labels, fit_X, mode='sil')
    cal_val = clustering_in_eval(labels, fit_X, mode='cal')

    # extrinsic evaluation, entropy
    label_tuples = out_eval_prep(ted_df, labels)
    entropy = clustering_out_eval(label_tuples)

    # record metric
    metric_record(results_csv_path, cluster, sil_val, cal_val,
                  n_clusters=n_clusters,
                  eps=eps,
                  linkage=linkage,
                  min_samples=min_samples,
                  feature=feature,
                  preprocess=preprocess,
                  lsa_n=lsa_n,
                  entropy=entropy,
                  )

    # analysis word distribution and word cloud
    if is_wordcloud:
        wordcloud_analysis(cluster, output_dir, labels, tf_matrix, idf_arr, words, ted_df)

    # show tsne result when LSA is applied
    if tsne:
        tsne_save_path = os.path.join(output_dir, '{}_tsne'.format(cluster))
        tsne_plot(labels, fit_X, save_path=tsne_save_path)


if __name__ == '__main__':
    main()
