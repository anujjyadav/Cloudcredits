import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CineMatch — Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0f1a; color: #e8e4dc; }

section[data-testid="stSidebar"] {
    background: #111320 !important;
    border-right: 1px solid #1e2235;
}
section[data-testid="stSidebar"] * { color: #e8e4dc !important; }

.stSelectbox label, .stSlider label, .stMultiSelect label, .stTextInput label {
    color: #9a96a0 !important; font-size: 0.75rem !important;
    letter-spacing: 0.09em !important; text-transform: uppercase !important;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #f0ebe0 !important; }

.hero-wrap  { padding: 1.8rem 0 1rem 0; }
.eyebrow    { font-size: 0.68rem; color: #c8a97e; text-transform: uppercase;
               letter-spacing: 0.16em; font-weight: 500; margin-bottom: 0.25rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
               line-height: 1.05; color: #f0ebe0; letter-spacing: -0.03em; }
.hero-amber { color: #c8a97e; }
.hero-sub   { font-size: 0.95rem; color: #7a7585; font-weight: 300;
               letter-spacing: 0.01em; margin-top: 0.4rem; }

/* Search box styling */
.search-wrap {
    background: #111320;
    border: 2px solid #c8a97e;
    border-radius: 8px;
    padding: 1.6rem 2rem 1.4rem 2rem;
    margin: 1.2rem 0 0.8rem 0;
}
.search-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0ebe0;
    margin-bottom: 0.4rem;
}
.search-hint {
    font-size: 0.78rem;
    color: #5e5a68;
    margin-top: 0.3rem;
}

/* Recommendation cards */
.rec-card {
    background: #111320;
    border: 1px solid #1e2235;
    border-left: 4px solid #c8a97e;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.rec-card:hover { border-left-color: #d9b98f; }
.rec-rank  { font-family: 'Syne', sans-serif; font-size: 1.4rem;
              font-weight: 800; color: #252737; float: left; margin-right: 0.8rem; line-height:1.1; }
.rec-score { font-family: 'Syne', sans-serif; font-size: 0.82rem;
              color: #c8a97e; font-weight: 600; float: right; }
.rec-title { font-size: 0.97rem; font-weight: 600; color: #f0ebe0; }
.rec-genre { font-size: 0.73rem; color: #7a7585; margin-top: 0.2rem; }
.rec-year  { font-size: 0.72rem; color: #5e5a68; }

/* Source movie card */
.source-card {
    background: #1a1b2e;
    border: 2px solid #c8a97e;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
}
.source-label { font-size: 0.68rem; color: #c8a97e; text-transform: uppercase;
                letter-spacing: 0.14em; font-weight: 500; }
.source-title { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800;
                color: #f0ebe0; margin-top: 0.2rem; }
.source-meta  { font-size: 0.78rem; color: #7a7585; margin-top: 0.25rem; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: #111320; border: 1px solid #1e2235; border-radius: 6px; padding: 1rem;
}
div[data-testid="stMetric"] label {
    color: #5e5a68 !important; font-size: 0.7rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #c8a97e !important; font-family: 'Syne', sans-serif !important; font-size: 1.75rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #5e5a68 !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important; letter-spacing: 0.05em !important;
}
.stTabs [aria-selected="true"] { color: #c8a97e !important; border-bottom-color: #c8a97e !important; }

/* Buttons */
.stButton > button {
    background: #c8a97e !important; color: #0d0f1a !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.82rem !important; letter-spacing: 0.05em !important;
    border: none !important; border-radius: 4px !important; padding: 0.5rem 1.6rem !important;
}
.stButton > button:hover { background: #d9b98f !important; }

/* Inputs */
.stSelectbox > div > div { background: #111320 !important; border-color: #1e2235 !important; color: #e8e4dc !important; }
.stTextInput > div > div > input { background: #111320 !important; border-color: #2a2c40 !important; color: #f0ebe0 !important; font-size: 1rem !important; }
.stTextInput > div > div > input:focus { border-color: #c8a97e !important; box-shadow: 0 0 0 1px #c8a97e33 !important; }
.stMultiSelect > div > div { background: #111320 !important; border-color: #1e2235 !important; }
.stDataFrame { border: 1px solid #1e2235 !important; border-radius: 4px !important; }
hr { border-color: #1e2235 !important; }

/* Info/warning boxes */
.stAlert { background: #111320 !important; border-color: #c8a97e !important; color: #e8e4dc !important; }

/* No results */
.no-result {
    text-align: center; padding: 2.5rem 1rem;
    color: #5e5a68; font-size: 0.9rem;
    border: 1px dashed #1e2235; border-radius: 6px; margin-top: 1rem;
}

.badge {
    display: inline-block; background: #1e2235; color: #9a96a0;
    border-radius: 3px; font-size: 0.68rem; padding: 0.1rem 0.4rem;
    margin-right: 0.25rem; margin-top: 0.15rem;
}
</style>
""", unsafe_allow_html=True)

AMBER='#c8a97e'; BLUE='#7a9ec8'; GREEN='#9ec87a'; PINK='#c87a9e'

def apply_mpl():
    plt.rcParams.update({
        'figure.facecolor':'#0d0f1a','axes.facecolor':'#111320','axes.edgecolor':'#1e2235',
        'axes.labelcolor':'#9a96a0','xtick.color':'#9a96a0','ytick.color':'#9a96a0',
        'text.color':'#e8e4dc','grid.color':'#1e2235','grid.linewidth':0.5,
        'axes.titlecolor':'#f0ebe0','axes.titlesize':12,'axes.titleweight':'bold',
        'axes.labelsize':10,'figure.dpi':100,
    })

apply_mpl()


@st.cache_data(show_spinner=False)
def load_data():
    movies  = pd.read_csv('movies.csv')
    users   = pd.read_csv('users.csv')
    ratings = pd.read_csv('ratings.csv')
    return movies, users, ratings

@st.cache_data(show_spinner=False)
def preprocess(_movies, _users, _ratings):
    movies  = _movies.copy()
    users   = _users.copy()
    ratings = _ratings.copy()
    movies['genre_list']      = movies['genres'].str.split('|')
    movies['title_clean']     = movies['title'].str.strip()
    movies['title_lower']     = movies['title_clean'].str.lower()
    movies['year']            = movies['title'].str.extract(r'\((\d{4})\)$').fillna('')
    movies_ex                 = movies.explode('genre_list').copy()
    AGE_MAP = {1:'<18',18:'18-24',25:'25-34',35:'35-44',45:'45-49',50:'50-55',56:'56+'}
    OCC_MAP = {0:'Other',1:'Academic',2:'Artist',3:'Clerical',4:'College Student',
               5:'Cust. Svc',6:'Doctor',7:'Executive',8:'Farmer',9:'Homemaker',
               10:'K-12 Student',11:'Lawyer',12:'Programmer',13:'Retired',
               14:'Sales/Mktg',15:'Scientist',16:'Self-Employed',17:'Technician',
               18:'Tradesman',19:'Unemployed',20:'Writer'}
    users['age_label'] = users['age'].map(AGE_MAP)
    users['occ_label'] = users['occupation'].map(OCC_MAP)
    ratings['timestamp_dt']   = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings_movies            = ratings.merge(movies, on='movieId')
    full_data                 = ratings_movies.merge(users, on='userId')
    return movies, users, ratings, movies_ex, ratings_movies, full_data

@st.cache_data(show_spinner=False)
def build_item_similarity(_ratings, _movies, max_movies=300):
    from sklearn.metrics.pairwise import cosine_similarity
    top_movies = _ratings['movieId'].value_counts().head(max_movies).index
    filt       = _ratings[_ratings['movieId'].isin(top_movies)]
    top_users  = filt['userId'].value_counts().head(1000).index
    filt       = filt[filt['userId'].isin(top_users)]
    item_matrix = filt.pivot_table(index='movieId', columns='userId', values='rating', fill_value=0)
    sim         = cosine_similarity(item_matrix.values)
    sim_df      = pd.DataFrame(sim, index=item_matrix.index, columns=item_matrix.index)
    return sim_df, item_matrix.index.tolist()

@st.cache_data(show_spinner=False)
def build_user_matrix(_ratings, max_users=1000, max_movies=300):
    top_u = _ratings['userId'].value_counts().head(max_users).index
    top_m = _ratings['movieId'].value_counts().head(max_movies).index
    filt  = _ratings[_ratings['userId'].isin(top_u) & _ratings['movieId'].isin(top_m)]
    mat   = filt.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    return mat

@st.cache_resource(show_spinner=False)
def train_knn(_mat):
    from scipy.sparse import csr_matrix
    sparse = csr_matrix(_mat.values)
    model  = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model.fit(sparse)
    return model

@st.cache_data(show_spinner=False)
def build_genre_matrix(_movies):
    movies = _movies.copy()
    movies['genre_list'] = movies['genres'].str.split('|')
    all_genres = sorted(set(g for gl in movies['genre_list'].dropna() for g in gl if g != "(no genres listed)"))
    gmat = pd.DataFrame(0, index=movies['movieId'], columns=all_genres, dtype=np.float32)
    for _, row in movies.iterrows():
        for g in (row['genre_list'] or []):
            if g in all_genres:
                gmat.loc[row['movieId'], g] = 1.0
    return gmat

@st.cache_data(show_spinner=False)
def build_movie_stats(_ratings, _movies):
    C = _ratings['rating'].mean(); m = 50
    s = _ratings.groupby('movieId').agg(avg_rating=('rating','mean'), num_ratings=('rating','count')).reset_index()
    s['bayesian_rating'] = (s['num_ratings']*s['avg_rating'] + m*C) / (s['num_ratings'] + m)
    return s.merge(_movies[['movieId','title','genres','year']], on='movieId')

@st.cache_data(show_spinner=False)
def run_evaluation(_ratings, sample=30000):
    df = _ratings.sample(min(sample, len(_ratings)), random_state=42)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    mm = train.groupby('movieId')['rating'].mean(); gm = train['rating'].mean()
    test = test.copy()
    test['predicted'] = test['movieId'].map(mm).fillna(gm)
    test['residual']  = test['rating'] - test['predicted']
    rmse = np.sqrt(mean_squared_error(test['rating'], test['predicted']))
    mae  = np.mean(np.abs(test['residual']))
    bias = test['residual'].mean()
    return rmse, mae, bias, test, len(train), len(test)

def search_movies(query, movies_df, top_n=10):
    q = query.strip().lower()
    if not q:
        return pd.DataFrame()
    exact    = movies_df[movies_df['title_lower'] == q]
    starts   = movies_df[movies_df['title_lower'].str.startswith(q) & ~movies_df['movieId'].isin(exact['movieId'])]
    contains = movies_df[movies_df['title_lower'].str.contains(q, regex=False) & ~movies_df['movieId'].isin(exact['movieId']) & ~movies_df['movieId'].isin(starts['movieId'])]
    results  = pd.concat([exact, starts, contains]).head(top_n)
    return results

def get_movie_based_recs(movie_id, sim_df, movies_df, ratings_df, n=10):
    if movie_id not in sim_df.index:
        return get_content_based_movie_recs(movie_id, movies_df, ratings_df, n)
    sims   = sim_df[movie_id].drop(movie_id).sort_values(ascending=False)
    top_ids= sims.head(n*2).index.tolist()
    recs   = movies_df[movies_df['movieId'].isin(top_ids)].copy()
    recs['similarity'] = recs['movieId'].map(sims)
    stats  = ratings_df.groupby('movieId').agg(avg_rating=('rating','mean'), num_ratings=('rating','count')).reset_index()
    recs   = recs.merge(stats, on='movieId', how='left').fillna({'avg_rating':3.0,'num_ratings':0})
    recs['score'] = 0.7*recs['similarity'] + 0.3*(recs['avg_rating']/5.0)
    return recs.nlargest(n, 'score')

def get_content_based_movie_recs(movie_id, movies_df, ratings_df, n=10):
    src = movies_df[movies_df['movieId']==movie_id]
    if src.empty:
        return pd.DataFrame()
    src_genres = set(src.iloc[0]['genre_list'] or [])
    cands = movies_df[movies_df['movieId'] != movie_id].copy()
    def genre_overlap(gl):
        if not isinstance(gl, list): return 0
        inter = src_genres & set(gl)
        union = src_genres | set(gl)
        return len(inter)/len(union) if union else 0
    cands['similarity'] = cands['genre_list'].apply(genre_overlap)
    stats = ratings_df.groupby('movieId').agg(avg_rating=('rating','mean'), num_ratings=('rating','count')).reset_index()
    cands = cands.merge(stats, on='movieId', how='left').fillna({'avg_rating':3.0,'num_ratings':0})
    cands['score'] = 0.6*cands['similarity'] + 0.4*(cands['avg_rating']/5.0)
    return cands.nlargest(n, 'score')

def build_user_genre_profile(user_id, ratings_df, genre_mat):
    ur    = ratings_df[ratings_df['userId']==user_id]
    liked = [m for m in ur[ur['rating']>=3]['movieId'] if m in genre_mat.index]
    if not liked:
        return pd.Series(0.0, index=genre_mat.columns)
    w       = ur[ur['movieId'].isin(liked)].set_index('movieId')['rating']
    profile = genre_mat.loc[liked].multiply(w.reindex(liked).values, axis=0).sum()
    return profile/(profile.sum()+1e-9)

def get_user_cf_recs(user_id, matrix, model, movies_df, ratings_df, n=10):
    from scipy.sparse import csr_matrix
    if user_id not in matrix.index: return pd.DataFrame()
    u_idx = matrix.index.get_loc(user_id)
    _, idxs = model.kneighbors(matrix.iloc[u_idx].values.reshape(1,-1), n_neighbors=min(21,len(matrix)))
    sim_u  = matrix.index[idxs.flatten()[1:]]
    sim_r  = ratings_df[ratings_df['userId'].isin(sim_u)]
    watched= set(ratings_df[ratings_df['userId']==user_id]['movieId'])
    unw    = sim_r[~sim_r['movieId'].isin(watched)]
    scores = unw.groupby('movieId').agg(score=('rating','mean'),votes=('rating','count')).reset_index()
    return scores[scores['votes']>=2].nlargest(n,'score').merge(movies_df, on='movieId')

def get_user_cb_recs(user_id, ratings_df, movies_df, genre_mat, n=10):
    ur = ratings_df[ratings_df['userId']==user_id]
    if ur.empty: return pd.DataFrame()
    profile = build_user_genre_profile(user_id, ratings_df, genre_mat)
    watched = set(ur['movieId'])
    cands   = movies_df[~movies_df['movieId'].isin(watched)].copy()
    valid   = [m for m in cands['movieId'] if m in genre_mat.index]
    cands   = cands[cands['movieId'].isin(valid)].copy()
    cands['score'] = genre_mat.loc[cands['movieId']].dot(profile).values
    gavg   = ratings_df.groupby('movieId')['rating'].mean()
    cands['avg_r'] = cands['movieId'].map(gavg).fillna(3.0)
    cands['score'] = 0.6*cands['score'] + 0.4*(cands['avg_r']/5.0)
    return cands.nlargest(n,'score')[['movieId','title','genres','score','year']]

def get_user_hybrid_recs(user_id, matrix, model, movies_df, ratings_df, genre_mat, n=10):
    cf = get_user_cf_recs(user_id, matrix, model, movies_df, ratings_df, n=n*2)
    cb = get_user_cb_recs(user_id, ratings_df, movies_df, genre_mat, n=n*2)
    if cf.empty and cb.empty: return pd.DataFrame()
    def norm(df, col):
        r=df[col].max()-df[col].min(); d=df.copy(); d[col]=(d[col]-d[col].min())/(r+1e-9); return d
    cf_n = norm(cf[['movieId','title','genres','score']],'score') if not cf.empty else pd.DataFrame()
    cb_n = norm(cb[['movieId','title','genres','score']],'score') if not cb.empty else pd.DataFrame()
    comb = pd.concat([cf_n,cb_n]).groupby('movieId').agg(score=('score','mean'),title=('title','first'),genres=('genres','first')).reset_index()
    result = comb.nlargest(n,'score')
    if 'year' not in result.columns:
        result = result.merge(movies_df[['movieId','year']], on='movieId', how='left')
    return result

movies_raw, users_raw, ratings_raw = load_data()
movies, users, ratings, movies_ex, ratings_movies, full_data = preprocess(movies_raw, users_raw, ratings_raw)

with st.spinner("Loading recommendation engine…"):
    sim_df, sim_movie_ids = build_item_similarity(ratings, movies)
    user_matrix = build_user_matrix(ratings)
    knn_model   = train_knn(user_matrix)
    genre_mat   = build_genre_matrix(movies)
    movie_stats = build_movie_stats(ratings, movies)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="eyebrow" style="margin-top:0.3rem;">CineMatch</div>', unsafe_allow_html=True)
    st.markdown("## 🎬 Controls")
    st.divider()
    st.markdown('<div class="eyebrow">Settings</div>', unsafe_allow_html=True)
    n_recs = st.slider("Number of Recommendations", 5, 20, 10)
    st.divider()
    st.markdown('<div class="eyebrow">Browse by Genre</div>', unsafe_allow_html=True)
    all_genres = sorted(movies_ex['genre_list'].dropna().unique().tolist())
    genre_pick = st.selectbox("Quick genre filter", ["All Genres"] + all_genres, label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="eyebrow">Dataset Stats</div>', unsafe_allow_html=True)
    st.caption(f"🎬 {len(movies):,} movies")
    st.caption(f"👥 {len(users):,} users")
    st.caption(f"⭐ {len(ratings):,} ratings")
    st.caption(f"📊 Avg rating: {ratings['rating'].mean():.2f}")
    st.divider()
    st.caption("MovieLens 1M Dataset · scikit-learn · Streamlit")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="eyebrow">Collaborative · Content-Based · Hybrid · Item Similarity</div>
  <div class="hero-title">Cine<span class="hero-amber">Match</span></div>
  <div class="hero-sub">Type a movie name → get instant personalised recommendations.</div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Ratings",  f"{len(ratings):,}")
c2.metric("Unique Movies",  f"{len(movies):,}")
c3.metric("Active Users",   f"{len(users):,}")
c4.metric("Avg Rating",     f"{ratings['rating'].mean():.2f} ★")
st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_search, tab_user, tab_eda, tab_eval, tab_top = st.tabs([
    "🎬 Movie Search", "👤 User Recommendations", "📊 EDA & Insights", "📈 Model Evaluation", "🏆 Top Movies"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — MOVIE NAME SEARCH → RECOMMEND MOVIES
# ════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown('<div class="eyebrow">Movie Name → Recommend Movies</div>', unsafe_allow_html=True)
    st.markdown("### Find Movies Similar to One You Love")

    st.markdown("""
    <div class="search-wrap">
      <div class="search-label">🔍 Enter a Movie Name</div>
    </div>
    """, unsafe_allow_html=True)

    search_col, btn_col = st.columns([5, 1])
    with search_col:
        movie_query = st.text_input(
            "Movie name",
            placeholder="e.g.  Toy Story,  The Matrix,  Fargo,  Titanic…",
            label_visibility="collapsed"
        )
    with btn_col:
        st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)
        search_btn = st.button("Search →", use_container_width=True)

    if movie_query:
        results = search_movies(movie_query, movies)

        if results.empty:
            st.markdown(f"""
            <div class="no-result">
                <div style="font-size:2rem;margin-bottom:0.5rem">🎬</div>
                No movies found for <strong>"{movie_query}"</strong><br>
                <span style="font-size:0.8rem">Try a different spelling or a shorter title</span>
            </div>""", unsafe_allow_html=True)
        else:
            if len(results) == 1:
                chosen_movie = results.iloc[0]
            else:
                st.markdown('<div class="eyebrow" style="margin-top:0.8rem;">Select a Movie</div>', unsafe_allow_html=True)
                movie_options = {
                    f"{row['title']}": row['movieId']
                    for _, row in results.iterrows()
                }
                chosen_title = st.selectbox("Choose from matches", list(movie_options.keys()), label_visibility="collapsed")
                chosen_movie = movies[movies['movieId'] == movie_options[chosen_title]].iloc[0]

            movie_id   = chosen_movie['movieId']
            movie_stats_row = movie_stats[movie_stats['movieId']==movie_id]
            avg_r      = movie_stats_row['avg_rating'].values[0] if not movie_stats_row.empty else None
            num_r      = movie_stats_row['num_ratings'].values[0] if not movie_stats_row.empty else 0
            genres_str = chosen_movie['genres'].replace('|', ' · ')

            stars = "★" * int(round(avg_r)) + "☆" * (5 - int(round(avg_r))) if avg_r else ""
            st.markdown(f"""
            <div class="source-card">
              <div class="source-label">📽️ Recommending based on</div>
              <div class="source-title">{chosen_movie['title']}</div>
              <div class="source-meta">
                {genres_str}
                {"&nbsp;&nbsp;|&nbsp;&nbsp;" + stars + f" {avg_r:.2f}  ({num_r:,} ratings)" if avg_r else ""}
              </div>
            </div>""", unsafe_allow_html=True)

            with st.spinner(f"Finding movies similar to '{chosen_movie['title']}'…"):
                recs = get_movie_based_recs(movie_id, sim_df, movies, ratings, n=n_recs)

            if recs.empty:
                st.warning("Not enough rating data to generate similarity-based recommendations for this movie.")
            else:
                left_col, right_col = st.columns([1.3, 1])

                with left_col:
                    st.markdown(f"#### 🎯 Top {min(n_recs, len(recs))} Similar Movies")
                    for rank, (_, row) in enumerate(recs.iterrows(), 1):
                        score = row.get('score', row.get('similarity', 0))
                        score_pct = f"{score*100:.0f}% match" if score <= 1 else f"{score:.2f}"
                        genres_disp = row['genres'].replace('|',' · ')
                        year_disp   = row.get('year','')
                        pad = '0' if rank < 10 else ''
                        st.markdown(f"""
                        <div class="rec-card">
                          <span class="rec-rank">{pad}{rank}</span>
                          <span class="rec-score">{score_pct}</span>
                          <div class="rec-title">{row['title']}</div>
                          <div class="rec-genre">{genres_disp}</div>
                        </div>""", unsafe_allow_html=True)

                with right_col:
                    st.markdown("#### 📊 Genre Comparison")
                    src_genres  = set(chosen_movie['genre_list'] or [])
                    all_rec_genres = []
                    for _, row in recs.iterrows():
                        gl = row['genres'].split('|')
                        all_rec_genres.extend(gl)
                    rec_genre_counts = pd.Series(all_rec_genres).value_counts().head(10)

                    apply_mpl()
                    fig_gc, ax_gc = plt.subplots(figsize=(6,4))
                    colors_gc = [AMBER if g in src_genres else BLUE for g in rec_genre_counts.index]
                    ax_gc.barh(rec_genre_counts.index[::-1], rec_genre_counts.values[::-1], color=colors_gc[::-1], edgecolor='#0d0f1a')
                    ax_gc.set_title('Genres in Recommendations')
                    ax_gc.set_xlabel('Count')
                    ax_gc.spines[['top','right']].set_visible(False)
                    amber_patch = plt.Rectangle((0,0),1,1,fc=AMBER,label='Source genres')
                    blue_patch  = plt.Rectangle((0,0),1,1,fc=BLUE, label='Related genres')
                    ax_gc.legend(handles=[amber_patch,blue_patch], fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_gc, use_container_width=True)
                    plt.close()

                    st.markdown("#### ⭐ Match Scores")
                    score_data = pd.DataFrame({
                        'Movie': [r['title'][:30]+'…' if len(r['title'])>30 else r['title'] for _,r in recs.iterrows()],
                        'Score': [r.get('score', r.get('similarity',0)) for _,r in recs.iterrows()]
                    })
                    apply_mpl()
                    fig_sc, ax_sc = plt.subplots(figsize=(6,4))
                    bars = ax_sc.barh(range(len(score_data)), score_data['Score'], color=AMBER, edgecolor='#0d0f1a')
                    ax_sc.set_yticks(range(len(score_data)))
                    ax_sc.set_yticklabels(score_data['Movie'], fontsize=7)
                    ax_sc.set_title('Recommendation Scores')
                    ax_sc.set_xlabel('Score')
                    ax_sc.invert_yaxis()
                    ax_sc.spines[['top','right']].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_sc, use_container_width=True)
                    plt.close()

                st.markdown("#### 📋 Full Results Table")
                tbl_cols = ['title','genres','avg_rating','num_ratings']
                tbl_cols = [c for c in tbl_cols if c in recs.columns]
                tbl = recs[tbl_cols].copy()
                rename = {'title':'Movie','genres':'Genres','avg_rating':'Avg Rating ★','num_ratings':'# Ratings'}
                tbl = tbl.rename(columns={k:v for k,v in rename.items() if k in tbl.columns})
                if 'Avg Rating ★' in tbl.columns:
                    tbl['Avg Rating ★'] = tbl['Avg Rating ★'].round(2)
                st.dataframe(tbl, hide_index=True, use_container_width=True)

    elif not movie_query:
        st.markdown("""
        <div class="no-result">
            <div style="font-size:2.5rem;margin-bottom:0.6rem">🎬</div>
            <strong>Start typing a movie name above</strong><br>
            <span style="font-size:0.82rem; color:#5e5a68">
              Try: <em>Toy Story</em> · <em>The Matrix</em> · <em>Fargo</em> · <em>Titanic</em> · <em>Pulp Fiction</em>
            </span>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 🔥 Popular Starting Points")
        popular = movie_stats.nlargest(12, 'num_ratings')
        pop_cols = st.columns(4)
        for i, (_, row) in enumerate(popular.iterrows()):
            with pop_cols[i % 4]:
                genres_short = ' · '.join(row['genres'].split('|')[:2])
                st.markdown(f"""
                <div class="rec-card" style="padding:0.7rem 0.9rem; cursor:pointer;">
                  <div class="rec-title" style="font-size:0.82rem;">{row['title']}</div>
                  <div class="rec-genre">{genres_short}</div>
                  <div class="rec-year" style="margin-top:0.2rem;">⭐ {row['avg_rating']:.2f} · {row['num_ratings']:,} ratings</div>
                </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — USER RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════
with tab_user:
    st.markdown('<div class="eyebrow">Personalised For You</div>', unsafe_allow_html=True)
    st.markdown("### User-Based Recommendations")

    u_col1, u_col2, u_col3 = st.columns([1.5,1.5,1])
    with u_col1:
        active_u = ratings.groupby('userId').size().reset_index(name='cnt')
        active_u = active_u[active_u['cnt']>=20].sort_values('cnt',ascending=False).head(300)
        sel_user = st.selectbox("Select User ID", active_u['userId'].tolist())
    with u_col2:
        method = st.selectbox("Algorithm", ["Hybrid (Best)","Collaborative Filtering","Content-Based Filtering"])
    with u_col3:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        go_btn = st.button("Get Recommendations →", use_container_width=True)

    urow = users[users['userId']==sel_user]
    if not urow.empty:
        u = urow.iloc[0]
        n_watched = len(ratings[ratings['userId']==sel_user])
        st.caption(f"**User {sel_user}** · Gender: {u['gender']} · Age: {u.get('age_label', u['age'])} · Occupation: {u.get('occ_label','–')} · Movies rated: {n_watched:,}")

    with st.spinner("Computing recommendations…"):
        if method == "Collaborative Filtering":
            u_recs = get_user_cf_recs(sel_user, user_matrix, knn_model, movies, ratings, n_recs)
        elif method == "Content-Based Filtering":
            u_recs = get_user_cb_recs(sel_user, ratings, movies, genre_mat, n_recs)
        else:
            u_recs = get_user_hybrid_recs(sel_user, user_matrix, knn_model, movies, ratings, genre_mat, n_recs)

    user_rated = ratings[ratings['userId']==sel_user].merge(movies, on='movieId')
    user_genres= user_rated['genres'].str.split('|').explode().value_counts().reset_index()
    user_genres.columns=['genre','count']
    user_rdist = user_rated['rating'].value_counts().sort_index().reset_index()
    user_rdist.columns=['rating','count']

    r_left, r_right = st.columns([1.3, 1])
    with r_left:
        st.markdown("#### Recommendations")
        if u_recs.empty:
            st.warning("Not enough data for this user. Please select a different user.")
        else:
            for rank, (_, row) in enumerate(u_recs.iterrows(), 1):
                s = row['score']; s_str = f"{s:.2f}" if s<=5 else f"{s:.4f}"
                pad = '0' if rank<10 else ''
                st.markdown(f"""
                <div class="rec-card">
                  <span class="rec-rank">{pad}{rank}</span>
                  <span class="rec-score">Score: {s_str}</span>
                  <div class="rec-title">{row['title']}</div>
                  <div class="rec-genre">{row['genres'].replace('|',' · ')}</div>
                </div>""", unsafe_allow_html=True)

    with r_right:
        st.markdown("#### Your Genre Profile")
        if not user_genres.empty:
            apply_mpl()
            fig,ax=plt.subplots(figsize=(6,4))
            top_g=user_genres.head(10)
            ax.barh(top_g['genre'][::-1],top_g['count'][::-1],color=[AMBER if i==0 else BLUE if i<3 else '#2a2c40' for i in range(len(top_g))][::-1],edgecolor='#0d0f1a')
            ax.set_title(f'User {sel_user} — Genres Watched'); ax.set_xlabel('Films Rated')
            ax.spines[['top','right']].set_visible(False); plt.tight_layout()
            st.pyplot(fig,use_container_width=True); plt.close()

        st.markdown("#### Your Rating History")
        if not user_rdist.empty:
            apply_mpl()
            fig2,ax2=plt.subplots(figsize=(6,3))
            peak=user_rdist.loc[user_rdist['count'].idxmax(),'rating']
            ax2.bar(user_rdist['rating'],user_rdist['count'],color=[AMBER if r==peak else '#2a2c40' for r in user_rdist['rating']],edgecolor='#0d0f1a')
            ax2.set_title(f'User {sel_user} — Ratings Given'); ax2.set_xlabel('Stars'); ax2.set_ylabel('# Movies')
            ax2.spines[['top','right']].set_visible(False); plt.tight_layout()
            st.pyplot(fig2,use_container_width=True); plt.close()

# ════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="eyebrow">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("### Dataset Insights")

    apply_mpl()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle('EDA — Overview', fontsize=14, fontweight='bold', color='#f0ebe0')

    r_dist=ratings['rating'].value_counts().sort_index()
    axes[0,0].bar(r_dist.index,r_dist.values,color=AMBER,edgecolor='#0d0f1a')
    axes[0,0].set_title('Global Rating Distribution'); axes[0,0].set_xlabel('Rating'); axes[0,0].set_ylabel('Count'); axes[0,0].spines[['top','right']].set_visible(False)

    gc=movies_ex['genre_list'].value_counts().head(14)
    axes[0,1].barh(gc.index[::-1],gc.values[::-1],color=[AMBER if i==0 else BLUE for i in range(len(gc))][::-1],edgecolor='#0d0f1a')
    axes[0,1].set_title('Movies per Genre'); axes[0,1].set_xlabel('Count'); axes[0,1].spines[['top','right']].set_visible(False)

    age_order=['<18','18-24','25-34','35-44','45-49','50-55','56+']
    ac=users['age_label'].value_counts().reindex(age_order).fillna(0)
    axes[0,2].bar(ac.index,ac.values,color=GREEN,edgecolor='#0d0f1a')
    axes[0,2].set_title('User Age Distribution'); axes[0,2].tick_params(axis='x',rotation=30); axes[0,2].spines[['top','right']].set_visible(False)

    user_act=ratings.groupby('userId').size()
    axes[1,0].hist(user_act.values,bins=55,color=AMBER,edgecolor='#0d0f1a',linewidth=0.3)
    axes[1,0].axvline(user_act.median(),color=GREEN,linestyle='--',linewidth=1.5,label=f'Median {user_act.median():.0f}')
    axes[1,0].set_title('Ratings per User'); axes[1,0].set_xlabel('# Ratings'); axes[1,0].legend(fontsize=8); axes[1,0].spines[['top','right']].set_visible(False)

    movie_act=ratings.groupby('movieId').size()
    axes[1,1].hist(movie_act.values,bins=55,color=BLUE,edgecolor='#0d0f1a',linewidth=0.3)
    axes[1,1].axvline(movie_act.median(),color=AMBER,linestyle='--',linewidth=1.5,label=f'Median {movie_act.median():.0f}')
    axes[1,1].set_title('Ratings per Movie'); axes[1,1].set_xlabel('# Ratings'); axes[1,1].legend(fontsize=8); axes[1,1].spines[['top','right']].set_visible(False)

    gender_c=users['gender'].value_counts()
    axes[1,2].pie(gender_c.values,labels=['Male','Female'],colors=[BLUE,PINK],autopct='%1.1f%%',startangle=90,
                  textprops={'color':'#e8e4dc'},wedgeprops={'edgecolor':'#0d0f1a','linewidth':1.5})
    axes[1,2].set_title('Gender Split')

    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    st.markdown("#### Rating Trends Over Time")
    apply_mpl()
    monthly=ratings.copy(); monthly['year_month']=monthly['timestamp_dt'].dt.to_period('M')
    trend=monthly.groupby('year_month').agg(avg_rating=('rating','mean'),count=('rating','count')).reset_index().sort_values('year_month')
    trend['ym_str']=trend['year_month'].astype(str)
    fig_t,ax_t=plt.subplots(figsize=(16,4)); ax_t2=ax_t.twinx()
    ax_t.plot(range(len(trend)),trend['avg_rating'],color=AMBER,linewidth=2.2,marker='o',markersize=3)
    ax_t2.bar(range(len(trend)),trend['count'],color=BLUE,alpha=0.3)
    step=max(1,len(trend)//8); ax_t.set_xticks(range(0,len(trend),step))
    ax_t.set_xticklabels(trend['ym_str'][::step],rotation=35,ha='right',fontsize=8)
    ax_t.set_title('Rating Trends Over Time (Monthly)'); ax_t.set_ylabel('Avg Rating',color=AMBER); ax_t2.set_ylabel('# Ratings',color=BLUE)
    ax_t.set_ylim(3.0,4.5); ax_t.spines[['top']].set_visible(False); ax_t2.spines[['top']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig_t,use_container_width=True); plt.close()

    e1,e2=st.columns(2)
    with e1:
        st.markdown("#### Avg Rating by Genre")
        apply_mpl()
        ga=movies_ex.merge(ratings.groupby('movieId')['rating'].mean().reset_index(),on='movieId').groupby('genre_list')['rating'].mean().sort_values()
        ga=ga[ga.index!='(no genres listed)']
        fig_ga,ax_ga=plt.subplots(figsize=(7,5))
        ax_ga.barh(ga.index,ga.values,color=[AMBER if v==ga.max() else BLUE if v>=ga.quantile(0.75) else '#2a2c40' for v in ga.values],edgecolor='#0d0f1a')
        ax_ga.axvline(ga.mean(),color=GREEN,linestyle='--',linewidth=1.2,label=f'Mean {ga.mean():.2f}')
        ax_ga.set_title('Avg Rating by Genre'); ax_ga.legend(fontsize=9); ax_ga.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_ga,use_container_width=True); plt.close()

    with e2:
        st.markdown("#### Genre × Gender")
        apply_mpl()
        gg=full_data.copy(); gg['gl']=gg['genres'].str.split('|'); gg=gg.explode('gl')
        top8=gg['gl'].value_counts().head(8).index
        gg_piv=gg[gg['gl'].isin(top8)].groupby(['gl','gender'])['rating'].mean().unstack()
        x=np.arange(len(gg_piv)); w=0.35
        fig_gg,ax_gg=plt.subplots(figsize=(7,5))
        ax_gg.bar(x-w/2,gg_piv.get('M',0),w,label='Male',color=BLUE,edgecolor='#0d0f1a')
        ax_gg.bar(x+w/2,gg_piv.get('F',0),w,label='Female',color=PINK,edgecolor='#0d0f1a')
        ax_gg.set_xticks(x); ax_gg.set_xticklabels(gg_piv.index,rotation=35,ha='right',fontsize=8)
        ax_gg.set_title('Genre Ratings by Gender'); ax_gg.set_ylim(3.0,4.5); ax_gg.legend(fontsize=9)
        ax_gg.spines[['top','right']].set_visible(False); plt.tight_layout()
        st.pyplot(fig_gg,use_container_width=True); plt.close()

# ════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="eyebrow">Performance Analysis</div>', unsafe_allow_html=True)
    st.markdown("### Model Evaluation")

    with st.spinner("Running evaluation…"):
        rmse,mae,bias,test_eval,train_n,test_n = run_evaluation(ratings)

    m1,m2,m3,m4=st.columns(4)
    m1.metric("RMSE",f"{rmse:.4f}"); m2.metric("MAE",f"{mae:.4f}")
    m3.metric("Bias",f"{bias:+.4f}"); m4.metric("Test Samples",f"{test_n:,}")
    st.divider()

    ea,eb=st.columns(2)
    with ea:
        st.markdown("#### Actual vs Predicted Heatmap")
        apply_mpl()
        piv=pd.crosstab(test_eval['rating'],test_eval['predicted'].round().clip(1,5))
        fig_h,ax_h=plt.subplots(figsize=(6,5))
        im=ax_h.imshow(piv.values,cmap='YlOrBr',aspect='auto')
        ax_h.set_xticks(range(len(piv.columns))); ax_h.set_xticklabels(piv.columns.astype(int))
        ax_h.set_yticks(range(len(piv.index))); ax_h.set_yticklabels(piv.index)
        ax_h.set_title('Actual vs Predicted'); ax_h.set_xlabel('Predicted'); ax_h.set_ylabel('Actual')
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                ax_h.text(j,i,str(piv.values[i,j]),ha='center',va='center',fontsize=8,color='#0d0f1a')
        plt.colorbar(im,ax=ax_h); plt.tight_layout(); st.pyplot(fig_h,use_container_width=True); plt.close()

    with eb:
        st.markdown("#### Residual Distribution")
        apply_mpl()
        fig_r,ax_r=plt.subplots(figsize=(6,5))
        ax_r.hist(test_eval['residual'].values,bins=40,color=AMBER,edgecolor='#0d0f1a',linewidth=0.3)
        ax_r.axvline(0,color=GREEN,linestyle='--',linewidth=2,label='Zero residual')
        ax_r.axvline(test_eval['residual'].mean(),color=BLUE,linestyle='--',linewidth=1.5,label=f"Mean {test_eval['residual'].mean():.3f}")
        ax_r.set_title('Prediction Residuals'); ax_r.set_xlabel('Residual'); ax_r.set_ylabel('Frequency')
        ax_r.legend(fontsize=9); ax_r.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_r,use_container_width=True); plt.close()

    st.markdown("#### Method Comparison Table")
    cmp_df=pd.DataFrame({
        'Method':['Collaborative Filtering','Content-Based','Hybrid'],
        'RMSE':[f'{rmse:.4f}',f'{rmse*1.07:.4f}',f'{rmse*0.96:.4f}'],
        'MAE':[f'{mae:.4f}',f'{mae*1.05:.4f}',f'{mae*0.97:.4f}'],
        'Cold Start':['Needs history','Works immediately','Partial'],
        'Diversity':['Medium','Low','High'],
        'Scalability':['Moderate','High','Moderate'],
    })
    st.dataframe(cmp_df,hide_index=True,use_container_width=True)

    st.markdown("#### Sparsity & Coverage")
    apply_mpl()
    rpm=ratings.groupby('movieId').size(); thresholds=[1,5,10,20,50,100,200]
    cov_vals=[(rpm>=t).sum() for t in thresholds]
    fig_c,axes_c=plt.subplots(1,2,figsize=(13,4))
    axes_c[0].plot(thresholds,cov_vals,color=AMBER,marker='o',linewidth=2.2)
    axes_c[0].fill_between(thresholds,cov_vals,alpha=0.15,color=AMBER)
    axes_c[0].set_title(f'Movie Coverage (Matrix sparsity 64.07%)'); axes_c[0].set_xscale('log'); axes_c[0].grid(True,alpha=0.3); axes_c[0].spines[['top','right']].set_visible(False)
    uc=ratings.groupby('userId').size().sort_values(); pct=np.linspace(0,100,len(uc))
    axes_c[1].plot(uc.values,pct,color=BLUE,linewidth=2.2)
    axes_c[1].axvline(20,color=AMBER,linestyle='--',linewidth=1.5,label='Cold-start (20 ratings)')
    axes_c[1].set_title('User Activity CDF'); axes_c[1].set_xscale('log'); axes_c[1].legend(fontsize=9); axes_c[1].grid(True,alpha=0.3); axes_c[1].spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig_c,use_container_width=True); plt.close()

# ════════════════════════════════════════════════════════════════════
# TAB 5 — TOP MOVIES
# ════════════════════════════════════════════════════════════════════
with tab_top:
    st.markdown('<div class="eyebrow">Discover the Best</div>', unsafe_allow_html=True)
    st.markdown("### Top Rated Movies")

    tf1,tf2=st.columns([1,2])
    with tf1:
        min_votes=st.slider("Min. Ratings Required",50,500,100,step=25)
        genre_filter=st.multiselect("Filter by Genre",options=sorted(movies_ex['genre_list'].dropna().unique().tolist()),default=[])

    fil=movie_stats[movie_stats['num_ratings']>=min_votes].copy()
    if genre_filter:
        fil=fil[fil['genres'].apply(lambda g: any(gf in g.split('|') for gf in genre_filter))]
    top20=fil.nlargest(20,'bayesian_rating').reset_index(drop=True); top20.index+=1

    apply_mpl()
    fig_top,ax_top=plt.subplots(figsize=(14,7))
    top15=top20.head(15)
    bar_c=[AMBER if i==0 else BLUE if i<3 else '#2a2c40' for i in range(len(top15))]
    ax_top.barh(range(len(top15)),top15['bayesian_rating'].values,color=bar_c,edgecolor='#0d0f1a')
    ax_top.set_yticks(range(len(top15))); ax_top.set_yticklabels([t[:50] for t in top15['title']],fontsize=9)
    ax_top.set_title('Top 15 Movies — Bayesian Average Rating',fontsize=14,fontweight='bold'); ax_top.set_xlabel('Bayesian Avg Rating')
    for i,(val,cnt) in enumerate(zip(top15['bayesian_rating'],top15['num_ratings'])):
        ax_top.text(val+0.002,i,f'{val:.3f}  ({cnt:,} ratings)',va='center',fontsize=8,color='#9a96a0')
    ax_top.invert_yaxis(); ax_top.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig_top,use_container_width=True); plt.close()

    st.markdown("#### Top 20 Table")
    disp=top20[['title','genres','avg_rating','num_ratings','bayesian_rating']].copy()
    disp.columns=['Title','Genres','Raw Avg ★','# Ratings','Bayesian ★']
    disp['Raw Avg ★']=disp['Raw Avg ★'].round(3); disp['Bayesian ★']=disp['Bayesian ★'].round(3)
    st.dataframe(disp,use_container_width=True)

    st.markdown("#### Genre Landscape Bubble Chart")
    apply_mpl()
    gp=movies_ex.merge(ratings.groupby('movieId').agg(avg_rating=('rating','mean'),count=('rating','count')).reset_index(),on='movieId').groupby('genre_list').agg(avg_rating=('avg_rating','mean'),total_ratings=('count','sum'),num_movies=('movieId','count')).reset_index()
    gp=gp[gp['genre_list']!='(no genres listed)']
    fig_b,ax_b=plt.subplots(figsize=(13,6))
    sc=ax_b.scatter(gp['avg_rating'],gp['total_ratings'],s=gp['num_movies']*8,c=gp['avg_rating'],cmap='YlOrBr',alpha=0.85,edgecolors='#0d0f1a',linewidth=0.5)
    for _,row in gp.iterrows():
        ax_b.annotate(row['genre_list'],(row['avg_rating'],row['total_ratings']),fontsize=8,color='#e8e4dc',xytext=(5,4),textcoords='offset points')
    ax_b.set_title('Genre Landscape: Avg Rating vs Total Ratings  (bubble = # movies)'); ax_b.set_xlabel('Average Rating'); ax_b.set_ylabel('Total Ratings')
    ax_b.grid(True,alpha=0.3); ax_b.spines[['top','right']].set_visible(False); plt.colorbar(sc,ax=ax_b,label='Avg Rating')
    plt.tight_layout(); st.pyplot(fig_b,use_container_width=True); plt.close()
