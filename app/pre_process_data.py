"""
prrprocess the TED csv
"""

import sys
import os
import pandas as pd
import ipdb

def main():
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ted_main_csv = os.path.join(top_dir, 'data', 'TED', 'ted_main.csv')
    ted_transcript_csv = os.path.join(top_dir, 'data', 'TED', 'transcripts.csv')
    new_csv_path = os.path.join(top_dir, 'data', 'TED', 'ted.csv')
    df_main = pd.read_csv(ted_main_csv)
    df_trans = pd.read_csv(ted_transcript_csv)
    df_main = df_main[['event', 'main_speaker',
                       'published_date', 'ratings', 'speaker_occupation', 'tags', 'title', 'url', 'views']]
    df_new = df_main.copy()
    for i, row in df_new.iterrows():
        url = row['url']
        try:
            transcript = df_trans[df_trans['url'] == url]['transcript'].values[0]
        except IndexError:
            print("no url found! {}".format(url))
            continue
        df_new.at[i, 'transcript'] = transcript

    df_new = df_new[pd.notnull(df_new['transcript'])]
    df_new = df_new.reset_index(drop=True)
    df_new['id'] = df_new.index
    df_new = df_new[['id', 'views', 'title', 'published_date', 'main_speaker', 'speaker_occupation', 'event', 'tags',
                     'ratings', 'transcript']]
    df_new.to_csv(new_csv_path, index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
