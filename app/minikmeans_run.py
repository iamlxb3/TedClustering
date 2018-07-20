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
from tqdm import tqdm

"""
python minikmeans_run.py
"""


def main():
    # set path
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data', 'TED')
    ted_path = os.path.join(data_dir, 'ted.csv')
    output_dir = os.path.join(top_dir, 'output')
    results_csv_path = os.path.join(top_dir, 'output', 'minikmeans_run_metrics.csv')

    # argument parse
    clear = False
    features= ['tf', 'tfidf']
    for feature in features:
        cluster = 'MiniBatchKMeans'
        preprocess = 'lsa'
        lsa_n_list = [x for x in range(50, 1000)][::50]
        is_wordcloud, linkage, eps, min_samples, tsne = None, None, None, None, None
        n_clusters = 10

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

        #
        for lsa_n in lsa_n_list:
            # pre process data & feature
            if preprocess == 'lsa':
                lsa = TruncatedSVD(n_components=lsa_n)
                if lsa_n > min(matrix.shape):
                    print("Warning, lsa n component won't be greated than max rank!")  # TODO, maybe?
                fit_X = lsa.fit_transform(matrix)
                print("Lsa done! Shape: {}".format(fit_X.shape))
            #

            parameter_dict = {
                'KMeans': (fit_X, n_clusters),
                'MiniBatchKMeans': (fit_X, n_clusters),
                'Spectral': (fit_X, n_clusters),
                'Agglomerative': (fit_X, n_clusters, linkage),
                'DBSCAN': (fit_X, n_clusters, eps, min_samples),
            }
            labels = clusters[cluster](*parameter_dict[cluster])

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
