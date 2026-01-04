import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# =======================
# CONFIG
# =======================
PARQUET_ID = "1vFRK_tCULQ7TvKCNmY8MzUR8Mi764I1z"
PARQUET_URL = f"https://drive.google.com/uc?id={PARQUET_ID}"
PARQUET_FILE = "tmdb_5000_movies.parquet"

OMDB_API_KEY = "3d8993f2"

# =======================
# STREAMLIT SETUP
# =======================
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")
st.write("Loading application… please wait")

# =======================
# LOAD DATA (PARQUET)
# =======================
@st.cache_resource
def load_data():
    if not os.path.exists(PARQUET_FILE):
        r = requests.get(PARQUET_URL, timeout=30)
        r.raise_for_status()
        with open(PARQUET_FILE, "wb") as f:
            f.write(r.content)
    return pd.read_parquet(PARQUET_FILE)

with st.spinner("Preparing recommendation engine…"):
    movies = load_data()

# =======================
# FINAL FEATURES (MATCH PARQUET SCHEMA)
# =======================
df = movies[["title", "tags"]]
df["tags"] = df["tags"].astype(str)

# =======================
# VECTORIZE + SIMILARITY
# =======================
@st.cache_resource
def build_similarity(df):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()
    return cosine_similarity(vectors)

with st.spinner("Computing similarity matrix…"):
    similarity = build_similarity(df)

# =======================
# POSTER FETCH (OMDB)
# =======================
@st.cache_data(show_spinner=False)
def fetch_poster(title):
    r = requests.get(
        "http://www.omdbapi.com/",
        params={"apikey": OMDB_API_KEY, "t": title},
        timeout=5
    ).json()
    if r.get("Poster") and r["Poster"] != "N/A":
        return r["Poster"]
    return None

# =======================
# RECOMMENDER
# =======================
def recommend(movie):
    idx = df[df["title"] == movie].index[0]
    distances = similarity[idx]

    recs = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names, posters = [], []
    for i in recs:
        title = df.iloc[i[0]].title
        names.append(title)
        posters.append(fetch_poster(title))
    return names, posters

# =======================
# UI
# =======================
movie = st.selectbox("Select a movie", df["title"].values)

if st.button("Recommend"):
    names, posters = recommend(movie)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            if posters[i]:
                st.image(posters[i], use_container_width=True)
            else:
                st.write("No poster")
            st.caption(names[i])
