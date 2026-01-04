import streamlit as st
import pandas as pd
import ast
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

PARQUET_ID = "1vFRK_tCULQ7TvKCNmY8MzUR8Mi764I1z"
PARQUET_URL = f"https://drive.google.com/uc?id={PARQUET_ID}"
PARQUET_FILE = "tmdb_5000_movies.parquet"

OMDB_API_KEY = "3d8993f2"

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")

@st.cache_resource
def load_data():
    if not os.path.exists(PARQUET_FILE):
        r = requests.get(PARQUET_URL, timeout=30)
        r.raise_for_status()
        with open(PARQUET_FILE, "wb") as f:
            f.write(r.content)
