"""
Extract vector of transcript (document - > vector)
"""

import sys
import os
import pandas as pd
import ipdb
import re
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from pywsd.utils import lemmatize_sentence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def to_tf(docs, save_path, is_normalize=True):
    """
    :param doc: document to bag of term frequencies
    :param normalize: normalize by the length of th doc
    :return:
        arr : M x N matrix (M is the number of documents, N is the number of words ( the size of the vocab)
        words: vocab
    """

    cv = CountVectorizer()
    cv_fit = cv.fit_transform(docs)
    words = cv.get_feature_names()
    arr = cv_fit.toarray()
    if is_normalize:
        arr = normalize(arr, axis=1)
    print("tf word dim: ", len(words))
    np.save(save_path, arr)
    print("Save tf arr to {}".format(save_path))
    return arr, words


def to_tfidf(docs, save_path, normalize='l2'):
    """
    :param doc: document to bag of tfidfs
    :param normalize: normalize mode
    :return:
        arr : M x N matrix (M is the number of documents, N is the number of words ( the size of the vocab)
        words: vocab
    """

    tfidf = TfidfVectorizer(norm=normalize)
    cv_fit = tfidf.fit_transform(docs)
    words = tfidf.get_feature_names()
    print("tfidf word dim: ", len(words))
    arr = cv_fit.toarray()
    np.save(save_path, arr)
    print("Save tfidf arr to {}".format(save_path))
    return arr, words

def extract_word(doc):
    """

    :param doc: a document
    :return: a document contains only words, no punctuation mark
    """
    return ' '.join(re.findall(r'\w+', doc))


def filter_stopwords(doc):
    """

    :param doc: a document, split by ' ' (space)
    :return: a document does not contain stopwords
    """
    doc_words = doc.split(' ')
    stop_words = set(stopwords.words('english'))
    new_doc = [word for word in doc_words if word not in stop_words]
    new_doc = ' '.join(new_doc)
    return new_doc


def lemmatize_docs(docs, verbose=False):
    """

    :param docs: list of documents
    :return: new_docs: list of lemmatized documents

    >>> text = 'Dew drops fall from the leaves'
    >>> lemmatize_sentence(text)
    >>> ['dew', 'drop', 'fall', 'from', 'the', 'leaf']
    """

    new_docs = []
    for i, doc in enumerate(docs):
        doc = extract_word(doc)
        words = lemmatize_sentence(doc)
        doc = ' '.join(words)
        doc = filter_stopwords(doc)
        new_docs.append(doc)
        if verbose:
            print("lemmatizing {}/{} done".format(i + 1, len(docs)))
    return new_docs


def main():
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ted_csv = os.path.join(top_dir, 'data', 'TED', 'ted.csv')
    process_doc_save_path = os.path.join(top_dir, 'data', 'TED', 'ted_clean.csv')

    df = pd.read_csv(ted_csv, encoding='utf-8')
    docs = df['transcript'].values
    processed_docs = lemmatize_docs(docs, verbose=True)

    df = {'transcript': [], 'id': []}
    for i, doc in enumerate(processed_docs):
        df['transcript'].append(doc)
        df['id'].append(i)
    df = pd.DataFrame(df)
    df.to_csv(process_doc_save_path, index=False, encoding='utf-8')
    print("cleaned ted transcript save to {}".format(process_doc_save_path))

def main2():
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ted_csv = os.path.join(top_dir, 'data', 'TED', 'ted_clean.csv')
    tf_arr_path = os.path.join(top_dir, 'data', 'TED', 'tf.npy')
    tfidf_arr_path = os.path.join(top_dir, 'data', 'TED', 'tfidf.npy')

    df = pd.read_csv(ted_csv, encoding='utf-8')
    docs = df['transcript'].values
    to_tf(docs, tf_arr_path)
    to_tfidf(docs, tfidf_arr_path)


if __name__ == '__main__':
    #main()
    main2()