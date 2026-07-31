# Hybrid Movie Recommendation System Based on Unsupervised Clustering and User Profiles
Unsupervised ML portfolio project on MovieLens 33M: PCA + K-Means clustering of movies by genre, then behavioral user profiling (diversity, generosit# 🎬 MovieLens Clustering — Movie Segmentation & User Behavior Profiles

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data--wrangling-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-portfolio%20project-success)

A portfolio project built on the **MovieLens 33M** dataset (*No Genomics* variant — title, genre, and ratings only, no genome-tag scores). The goal was to use **unsupervised learning** to group movies by genre similarity, then use those clusters to build **behavioral user profiles** that drive a simple rule-based recommendation strategy.

---

## 🗂️ About the data

- **Source:** [MovieLens 33M](https://grouplens.org/datasets/movielens/) — *No Genomics* variant
- **Files used:** `movies.csv` (title, year, genres) and `ratings.csv`
- **Temporal split:** movies released before **2015** go into training, everything from 2015 onward goes into a validation set — simulating the real-world scenario of "new movies arriving after the model was trained"

---

## 🧭 Project pipeline

```
movies.csv + ratings.csv
        │
        ▼
1. Genre parsing (MultiLabelBinarizer) + year extraction
        │
        ▼
2. PCA (10 components) on the binary genre matrix
        │
        ▼
3. K-Means (30 clusters) → each movie gets a "latent genre"
        │
        ▼
4. User behavior profile (diversity, generosity, frequency)
        │
        ▼
5. Segmentation into 8 user-profile octants
        │
        ▼
6. Per-cluster movie ranking + rule-based recommendation
```

---

## 1️⃣ Movie clustering (unsupervised)

Each movie has a list of genres (`Action|Comedy|Drama...`). This was one-hot encoded with `MultiLabelBinarizer`, reduced with **PCA** (10 components, retaining **83.6%** of the variance), and grouped with **K-Means**.

| Metric | Value |
|---|---|
| Number of clusters | 30 |
| Inertia (WCSS) | 10,869.83 |
| Silhouette Score | 0.6505 |
| Davies-Bouldin Index | 0.9460 |
| Calinski-Harabasz Score | 11,161.46 |

Validation movies (post-2015) were assigned to the already-trained clusters via `transform`/`predict` — no retraining, exactly as it would work in production.

<p align="center">
  <img width="815" height="625" alt="pca_clusters_2d" src="https://github.com/user-attachments/assets/31b9d10d-0484-46e5-b869-2a10434ec6a2" />
</p>

<p align="center"><i>2D PCA projection of the 30 movie clusters grouped by genre similarity.</i></p>

---

## 2️⃣ Building each user's profile

Once every movie belonged to a cluster, the next step was to turn each user's rating history into a **behavior profile**, computed across 3 axes:

- **Diversity** → how many different clusters a user tends to explore (normalized by the 30 clusters)
- **Generosity** → average rating given (z-score normalized per user, to remove the bias of "users who rate everything highly")
- **Frequency** → total number of movies rated, normalized

Based on the median of each axis, every user was classified into an **octant**:

`{Eclectic|Focused}` × `{Generous|Critical}` × `{Frequent|Casual}`

<p align="center">
 <img width="644" height="636" alt="perfil_3d_usuarios" src="https://github.com/user-attachments/assets/431bb99f-4470-4383-99b6-e8e433dd6edb" />
</p>

<p align="center"><i>3D distribution of users across the 3 behavioral axes.</i></p>

### Distribution of the profiles found

| Profile | Users |
|---|---:|
| Eclectic_Critical_Frequent | 32,937 |
| Eclectic_Generous_Casual | 31,519 |
| Eclectic_Critical_Casual | 28,166 |
| Eclectic_Generous_Frequent | 26,581 |
| Focused_Generous_Casual | 23,375 |
| Focused_Critical_Frequent | 22,846 |
| Focused_Generous_Frequent | 19,080 |
| Focused_Critical_Casual | 16,443 |
| **Total** | **~200,947** |

### Exploring the axes pairwise

<p align="center">
<img width="808" height="625" alt="perfil_frequencia_generosidade" src="https://github.com/user-attachments/assets/ba660e9c-e67b-4e1d-9e47-4963cdb9dcd7" />
<img width="808" height="625" alt="perfil_diversidade_generosidade" src="https://github.com/user-attachments/assets/5b25ba47-0644-472f-9105-f625f334e2d1" />
</p>
<p align="center">
<img width="808" height="625" alt="perfil_diversidade_frequencia" src="https://github.com/user-attachments/assets/ad2a571e-f668-4662-ab62-09a9b90ab0b3" />
<img width="1271" height="614" alt="boxplot_generosidade_por_num_clusters" src="https://github.com/user-attachments/assets/bfff168f-fb31-4e57-9f85-25ec98635d9f" />
</p>

---

## 3️⃣ Movie ranking & rule-based recommendation

With clusters and profiles in place, the final step was building a **per-cluster popularity ranking** (average rating, filtered to movies with 100+ ratings) and using each user's **octant** to decide the recommendation strategy:

- **Focused** → recommend only within the user's preferred clusters
- **Eclectic** → mix preferred clusters with movies from outside them (exploration)
- **Frequent** → gets a larger list (20 movies)
- **Casual** → gets a shorter list (10 movies)

This is a simple approach based on business rules over the behavioral segmentation — not a trained supervised model — but it already shows how unsupervised segmentation can feed directly into personalization.

---

## 🛠️ Tech stack

- **Python** (pandas, numpy)
- **scikit-learn** — `MultiLabelBinarizer`, `PCA`, `KMeans`, `StandardScaler`, `MinMaxScaler`, clustering metrics
- **Matplotlib / Plotly** — visualization
- **Joblib** — K-Means model persistence

---

## 📁 Structure

```
.
├── Notebook_ML_No_Genomics_MovieLens.ipynb   # full project notebook
├── kmeans_movies.pkl                          # saved K-Means model
└── images/                                    # charts generated during analysis
```

---

## 🚀 How to run

```bash
pip install pandas numpy scikit-learn matplotlib plotly joblib
```

Download the [MovieLens 33M (No Genomics)](https://grouplens.org/datasets/movielens/) dataset and point `DATA_PATH` in the notebook to the folder containing `movies.csv` and `ratings.csv`.

---

## 💭 Next steps

- Test different values of `k` and compare clustering metrics
- Explore clustering directly on user profiles too, not just on movies
- Replace the rule-based recommendation logic with an actual supervised model (that's the hook for the next portfolio project 👀)

---

<sub>Note on numbers: every metric and count in this README was pulled directly from the notebook's executed cell outputs (PCA variance, silhouette/Davies-Bouldin/Calinski-Harabasz scores, per-cluster movie counts, per-profile user counts). The only derived figure is the "~200,947" total, which is the sum of the 8 profile counts above.</sub>
y, frequency) to power a simple rule-based recommendation strategy.
