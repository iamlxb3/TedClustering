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
from sklearn.ensemble import IsolationForest
"""
python iforest.py -f tfidf --debug

"""


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', dest='clear', help='flag to clear the previous output',
                        default=False, action='store_true')
    parser.add_argument('-f', '--feature', dest='feature', help='valid features for clustering: tf, tfidf',
                        required=True, type=str, choices=('tf', 'tfidf'))
    parser.add_argument('--debug', action='store_true', help='debug_mode', default=False)
    parser.add_argument('--lsa_n', dest='lsa_n', help='n_components for latent semantic analysis',
                        type=int)
    parser.add_argument('--preprocess', dest='preprocess', help='data_preprocess',
                        type=str, choices=('lsa',))
    args = parser.parse_args()
    clear = args.clear
    feature = args.feature
    debug = args.debug
    preprocess = args.preprocess
    lsa_n = args.lsa_n
    return clear, feature, debug, preprocess, lsa_n


def main():
    # set path
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data', 'TED')
    ted_path = os.path.join(data_dir, 'ted.csv')
    output_dir = os.path.join(top_dir, 'output')

    # argument parse
    clear, feature, debug, preprocess, lsa_n = args_parse()

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

    #
    iforest = IsolationForest()
    iforest.fit(fit_X)
    scores = iforest.decision_function(fit_X)
    print("iforest done!")
    scores = [(i, score) for i, score in enumerate(scores)]
    scores = sorted(scores, key=lambda x:x[1])[0:10]
    ids = [x[0] for x in scores]
    titles = ted_df[ted_df['id'].isin(ids)]['title'].values
    for title, id_score in zip(titles, scores):
        id, score = id_score
        print("id: {}, score: {:.3f}, title: {}".format(id, score, title))
    #

if __name__ == '__main__':
    main()
