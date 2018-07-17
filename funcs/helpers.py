from sklearn import metrics
import re
import numpy as np
import collections
import pandas as pd
import os
from .wordcloud import wordcloud_generate

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
    if np.unique(labels).size == 1:
        print('labels: ', labels)
        return None
    if mode == 'sil':
        value = metrics.silhouette_score(X, labels, metric='euclidean')
        print("Silhouette Coefficient: ", value)
    elif mode == 'cal':
        value = metrics.calinski_harabaz_score(X, labels)
        print("Calinski-Harabaz: ", value)

    return value

def metric_record(results_csv_path, cluster, sil_val, cal_val,
                  eps=None,
                  n_clusters=None,
                  linkage=None,
                  min_samples=None,
                  feature=None,
                  preprocess=None):
    """
    Record the intrinsic metric
    :return:
    """

    valid_metrics = ('cluster', 'n_clusters', 'silhouette_coefficient', 'calinski_harabaz', 'eps',
                     'min_samples', 'linkage','feature','preprocess')
    results_df = {}
    for metric in valid_metrics:
        results_df[metric] = []

    results_df['cluster'].append(cluster)
    results_df['n_clusters'].append(n_clusters)
    results_df['silhouette_coefficient'].append(sil_val)
    results_df['calinski_harabaz'].append(cal_val)
    results_df['eps'].append(eps)
    results_df['min_samples'].append(min_samples)
    results_df['linkage'].append(linkage)
    results_df['feature'].append(feature)
    results_df['preprocess'].append(preprocess)

    results_df = pd.DataFrame(results_df)
    results_df = results_df[list(valid_metrics)]

    if os.path.isfile(results_csv_path):
        old_df = pd.read_csv(results_csv_path)
        print("Read existing csv from {}".format(results_csv_path))
        results_df = pd.concat([results_df, old_df], axis=0)
        results_df = results_df.drop_duplicates(keep='first')

    results_df.to_csv(results_csv_path, index=False)
    print("Record metric to {}".format(results_csv_path))
    return results_df

def wordcloud_analysis(cluster, output_dir, labels, tf_matrix, idf_arr, words, ted_df):
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