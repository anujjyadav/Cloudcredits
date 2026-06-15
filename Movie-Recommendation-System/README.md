
#  CineMatch — Movie Recommendation System

> **End-to-end ML pipeline** using the MovieLens 1M dataset to discover films personalised to every user's taste.

---

# Project Objective

Build an intelligent movie recommendation engine that predicts user preferences and surfaces relevant films from a catalogue of **3,883 movies** rated by **6,040 users** across **1,000,209 ratings**.

The system addresses three core challenges:
- **Personalisation** — recommend movies unique to each user's history
- **Cold-start resilience** — work even with limited rating data
- **Scalability** — handle millions of ratings efficiently

---

##  Dataset

**Source:** [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — GroupLens Research

| File | Rows | Columns | Description |
|---|---|---|---|
| `movies.csv` | 3,883 | 3 | `movieId`, `title`, `genres` |
| `users.csv` | 6,040 | 5 | `userId`, `gender`, `age`, `occupation`, `zip` |
| `ratings.csv` | 1,000,209 | 4 | `userId`, `movieId`, `rating` (1–5), `timestamp` |

**Key statistics:**
- Rating scale: 1–5 stars
- Date range: April 2000 – February 2003
- Average rating: 3.58 ★
- Matrix sparsity: 64.07%
- 18 unique genres

---

##  Algorithms Used

### 1. Collaborative Filtering (KNN)
Uses cosine similarity on the user-item rating matrix to find the most similar users, then recommends movies those neighbours rated highly that the target user hasn't seen.

- **Library:** `sklearn.neighbors.NearestNeighbors`
- **Similarity metric:** Cosine
- **Matrix:** 2,000 users × 500 movies (sparse)
- **Neighbours:** k = 20 (tuned)

### 2. Content-Based Filtering
Builds a weighted genre profile from each user's liked movies (rating ≥ 3), then scores unseen movies by genre overlap.

- **Feature vector:** 18-dimensional binary genre matrix
- **Scoring:** 60% genre similarity + 40% global average rating
- **Advantage:** Works immediately, no neighbours required

### 3. Hybrid (Weighted Average)
Combines normalised scores from both CF and CB:


hybrid_score = mean(norm(CF_score), norm(CB_score))


- Best overall accuracy and recommendation diversity
- Partial cold-start resilience



##  Evaluation Metrics

| Metric | Formula | CF | CB | Hybrid |
|---|---|---|---|---|
| **RMSE** | √(Σ(actual−pred)²/n) | 1.0289 | 1.1009 | **0.9878** |
| **MAE** | Σ\|actual−pred\|/n | 0.8167 | 0.8575 | **0.7922** |
| **Bias** | mean(pred−actual) | −0.0109 | −0.021 | −0.008 |
| **Coverage** | % test movies predicted | 93.2% | 100% | 100% |

> **Best model: Hybrid** — lowest RMSE (0.9878) and MAE (0.7922) with full coverage.

Train/Test split: **80% / 20%** stratified random split on 50,000 sampled ratings.

#  Results

### Final Model Performance (Hybrid)

| Metric | Value |
|---|---|
| RMSE | **0.9878** |
| MAE | **0.7922** |
| Bias | −0.008 |
| Train size | 40,000 |
| Test size | 10,000 |
| Coverage | 100% |

### Top 10 Recommended Movies (Sample User)
| Rank | Movie | Hybrid Score |
|---|---|---|
| 1 | Schindler's List (1993) | 0.8821 |
| 2 | Godfather, The (1972) | 0.8614 |
| 3 | Shawshank Redemption, The (1994) | 0.8592 |
| 4 | Usual Suspects, The (1995) | 0.8391 |
| 5 | Silence of the Lambs, The (1991) | 0.8274 |
| 6 | Fargo (1996) | 0.8103 |
| 7 | Remains of the Day, The (1993) | 0.7956 |
| 8 | Four Weddings and a Funeral (1994) | 0.7832 |
| 9 | Secrets & Lies (1996) | 0.7714 |
| 10 | Dr. Strangelove (1964) | 0.7601 |

### Hyperparameter Tuning (k-Neighbours)
| k | RMSE | MAE |
|---|---|---|
| 5 | 1.2389 | 0.9943 |
| 10 | 1.2598 | 0.9830 |
| 15 | 1.2617 | 0.9842 |
| **20** | **1.2354** | **0.9693** |
| 30 | 1.3246 | 1.0229 |
| 50 | 1.3388 | 1.0399 |

> Optimal k = **20** neighbours chosen for deployment.

### Top Genres by Average Rating
| Genre | Avg Rating |
|---|---|
| Film-Noir | 4.07 |
| Documentary | 3.93 |
| War | 3.89 |
| Drama | 3.77 |
| Crime | 3.69 |

---

##  Project Structure


cinématch/
├── movie_recommendation_system.ipynb   # Full 9-step ML pipeline notebook
├── streamlit_app.py                    # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── data/
    ├── movies.csv
    ├── users.csv
    └── ratings.csv
```

---

##  ML Pipeline (9 Steps)

| Step | Description |
|---|---|
| 1 | **Define the Problem** — Recommendation / regression objective |
| 2 | **Collect & Prepare Data** — Load CSVs, handle types, merge |
| 3 | **EDA** — Distributions, trends, genre analysis, gender breakdown |
| 4 | **Feature Engineering** — User-item matrix, genre vectors, Bayesian stats |
| 5 | **Split the Data** — 80/20 stratified train/test split |
| 6 | **Choose a Model** — CF, CB, Hybrid selection & justification |
| 7 | **Train the Model** — KNN fit, sample recommendations output |
| 8 | **Evaluate the Model** — RMSE, MAE, residual analysis, coverage |
| 9 | **Improve the Model** — k-tuning, method comparison, final selection |

---

##  Streamlit Web App

```bash
streamlit run streamlit_app.py
```

**Features:**
-  **Movie Name Search** — type any movie title, get instant similar recommendations
-  Personalised recommendations (CF / CB / Hybrid)
-  EDA & Insights dashboard
-  Model evaluation charts
-  Top Movies explorer with genre filter
- Dark cinema-themed UI

---

##  Quick Start

```bash
# 1. Clone / download the project
git clone https://github.com/your-username/cinematch.git
cd cinematch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset files in data/ folder
#    movies.csv, users.csv, ratings.csv

# 4. Run the Streamlit app
streamlit run streamlit_app.py

# 5. Open the Jupyter notebook (optional)
jupyter notebook movie_recommendation_system.ipynb
```

---

##  Requirements

See [`requirements.txt`](requirements.txt) for full dependency list.

**Core libraries:**
- `streamlit` — web application framework
- `pandas` / `numpy` — data manipulation
- `scikit-learn` — KNN, metrics, train/test split
- `scipy` — sparse matrix operations
- `matplotlib` / `seaborn` — visualisations

---

##  License

MIT License — free for educational and personal use.

---

## Author 

ANUJ YADAV

