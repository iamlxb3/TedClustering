# TedClustering

An attempt of using clusering algorithms to explore TED corpus.
The data is from https://www.kaggle.com/rounakbanik/ted-talks.

Used features: TF, TF-IDF, LSA

Clustering algorithms: K-means, MiniBatchK-means, hierarchical clustering, DBSCAN, iforest (abnormality detection)

| algorithm | entropy  |
| ------- | --- |
| MiniBatchKMeans | 5.06 |
| KMeans | 4.82 |
| hierarchical clustering average link | 5.28 |
| hierarchical clustering complete link | 5.17 |
| hierarchical clustering ward link | 4.85 |

Below shows how lsa affect the result
---

![alt text](https://cdn-images-1.medium.com/max/600/1*1aGZAyVS3kpJGcD_5_BNBQ.png)

Wordcloud for clusters 0-9
---
![alt text](https://cdn-images-1.medium.com/max/400/1*Hovxcg4MpFMQMcZhZmkmNw.png)
![alt text](https://cdn-images-1.medium.com/max/400/1*aiP_v3QKyu9ptuOabgV9Xw.png)
![alt text](https://cdn-images-1.medium.com/max/400/1*aiP_v3QKyu9ptuOabgV9Xw.png)
![alt text](https://cdn-images-1.medium.com/max/400/1*4VbtQWVlQPqs49R3-GhkpQ.png)
![alt text](https://cdn-images-1.medium.com/max/400/1*ODhgV37DINOmQG8TnUu8qg.png)
![alt text](https://cdn-images-1.medium.com/max/400/1*7xFL7aiQtnyrCwuqNycd5Q.png)
![alt text](https://cdn-images-1.medium.com/max/600/1*rtGp-b7JvsHRFtXmTyoT4A.png)
![alt text](https://cdn-images-1.medium.com/max/600/1*khnJGwch1s8g-3K0pzbFPA.png)
![alt text](https://cdn-images-1.medium.com/max/600/1*CJVGZ9yYNZWbE5_VPEcVFg.png)
![alt text](https://cdn-images-1.medium.com/max/600/1*_GrlyHxR2C4isbCmgq6hjw.png)

Tsne Project of clusters 0-9
---
![alt text](https://cdn-images-1.medium.com/max/800/1*1gH3h61rVwMYLCoBzezQVw.png)

Abnormality detection by iforest ( the most distinctive Ted talks), TFIDF + LSA
---

| score | tile  |
| ------- | --- |
| -0.039 | An 8-dimensional model of the universe |
| -0.038 | Debate: Does the world need nuclear energy? |
| -0.025 | Does democracy stifle economic growth? |
| -0.021 | Why bees are disappearing |
| -0.018 | How we're growing baby corals to rebuild reefs |
| -0.015 | Our refugee system is failing. Here's how we can fix it |
| -0.012 | The laws that sex workers really want |
| -0.009 | How fear of nuclear power is hurting the environment |
| -0.008 | The refugee crisis is a test of our character |
| -0.007 | Why I still have hope for coral reefs |
