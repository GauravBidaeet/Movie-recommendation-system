import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

OMDB_API_KEY = "3d8993f2"

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv", encoding="latin-1")
    credits = pd.read_csv("tmdb_5000_credits.csv", encoding="latin-1")
    return movies.merge(credits, on="title")

movies = load_data()

def convert(obj):
    return [i["name"] for i in ast.literal_eval(obj)]

def convert_cast(obj):
    return [i["name"] for i in ast.literal_eval(obj)[:3]]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            return [i["name"]]
    return []

movies = movies[[
    "movie_id","title","overview","genres","keywords","cast","crew"
]]
movies.dropna(inplace=True)

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["crew"] = movies["crew"].apply(fetch_director)

for col in ["genres","keywords","cast","crew"]:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

movies["overview"] = movies["overview"].apply(lambda x: x.split())

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

df = movies[["movie_id","title","tags"]]
df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()
similarity = cosine_similarity(vectors)

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
