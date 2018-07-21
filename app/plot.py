import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    csv_path1 = r'D:\我的坚果云\Github\TED_clustering\output\minikmeans_run_metrics.csv'
    df1 = pd.read_csv(csv_path1)

    # plot LDA
    df1 = df1[df1['feature'] == 'tfidf']
    sns.set(style="darkgrid")
    ax = sns.pointplot(x="lsa_n", y="entropy", data=df1)
    plt.show()
    #

    # compare all
    #

    # K-means vs mini-batch
    #

    # Agglomerative

    #