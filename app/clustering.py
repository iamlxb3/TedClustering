import sys
import os
import re
import ipdb
import pandas as pd
import argparse
import numpy as np
import collections
import shutil
from sklearn.cluster import KMeans
from sklearn import metrics

sys.path.append('..')
from funcs.wordcloud import wordcloud_generate


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', dest='clear', help='flag to clear the previous output',
                        default=False, action='store_true')
    parser.add_argument('-f', '--feature', dest='feature', help='valid features for clustering: tf, tfidf',
                        required=True, type=str, metavar='tf/tfidf')
    parser.add_argument('-c', '--cluster', dest='cluster', help='valid clusters for clustering: kmeans',
                        required=True, type=str, metavar='kmeans')

    args = parser.parse_args()
    cluster = args.cluster
    feature = args.feature
    clear = args.clear
    return feature, cluster, clear


def k_means(arr, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=0, max_iter=10)
    result = model.fit(arr)
    labels = result.labels_
    print("K_means clustering done!")
    return labels


def results_analysis(tf_matrix, idf_arr, indexes, words, ted_df):
    """
    Analysis clustering results
    :return:
    """
    # get the words for each cluster
    tfidfs = tf_matrix[indexes]
    tfidfs = np.sum(tfidfs, axis=0)
    assert tfidfs.ndim == 1
    assert idf_arr.ndim == 1
    tfidfs = tfidfs * idf_arr
    word_tfidf = zip(words, tfidfs)
    word_tfidf = [(word, tfidf) for word, tfidf in word_tfidf if tfidf > 0]
    word_tfidf = dict(word_tfidf)
    #

    # get the meta info
    ted_df = ted_df[ted_df['id'].isin(indexes)]
    meta_dict = {}
    views = sum(ted_df['views'].values)
    title = tuple(ted_df['title'].values)
    published_date = tuple(ted_df['published_date'].values)
    main_speaker = tuple(ted_df['main_speaker'].values)
    speaker_occupation = tuple(ted_df['speaker_occupation'].values)
    event = ted_df['event'].values
    tags = ted_df['tags'].values
    tags = [word for tag in tags for word in re.findall(r'\w+', tag)]
    ratings = ted_df['ratings'].values
    ratings_dict = collections.defaultdict(lambda: 0)
    for rating in ratings:
        rating = eval(rating)
        for dict_ in rating:
            ratings_dict[dict_['name']] += dict_['count']
    ratings = tuple(ratings_dict.items())
    meta_dict['ratings'] = ratings
    meta_dict['tags'] = tags
    meta_dict['event'] = event
    meta_dict['speaker_occupation'] = speaker_occupation
    meta_dict['main_speaker'] = main_speaker
    meta_dict['published_date'] = published_date
    meta_dict['title'] = title
    meta_dict['views'] = views

    return word_tfidf, meta_dict


def clustering_eval(labels, X, mode='sil'):
    """
    :param mode:
           sil: Silhouette Coefficient
           cal: Calinski-Harabaz
    :param labels: clustering labels
    :param X: M x N Document-feature maxtrix
    :return:
    """
    if mode == 'sil':
        value = metrics.silhouette_score(X, labels, metric='euclidean')
        print("Silhouette Coefficient: ", value)
    elif mode == 'cal':
        value = metrics.calinski_harabaz_score(X, labels)
        print("Calinski-Harabaz: ", value)


def main():
    # set path
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data', 'TED')
    ted_path = os.path.join(data_dir, 'ted.csv')
    output_dir = os.path.join(top_dir, 'output')

    # argument parse
    feature, cluster, clear = args_parse()

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
    clustering = {'kmeans': k_means}
    n_clusters = 4
    X = matrix[0:20]  # TODO
    labels = clustering[cluster](X, n_clusters)

    # intrinsic evaluation
    clustering_eval(labels, X, mode='sil')
    clustering_eval(labels, X, mode='cal')

    # analysis results
    labels_dict = collections.Counter(labels)
    print(labels_dict)
    for key in labels_dict.keys():
        indexes = np.where(labels == key)[0]
        word_tfidf, meta_dict = results_analysis(tf_matrix, idf_arr, indexes, words, ted_df)

        # makedir for one key
        save_dir = os.path.join(output_dir, "{}_cluster_{}".format(cluster, key))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            print("Make new dir {}".format(save_dir))
        save_path = os.path.join(save_dir, "transcript_wordcloud.png")

        wordcloud_generate(word_tfidf, save_path=save_path, is_show=True)
        break


if __name__ == '__main__':
    main()
