import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "cc942ffa68d1785d8668455665594f73"
TMDB_URL = "https://api.themoviedb.org/3/movie/{}?api_key={}"

@st.cache_data
def load_data_and_compute_similarity():
    # Load movie data
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)

    # Vectorize tags
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()

    # Compute cosine similarity
    similarity = cosine_similarity(vectors)
    return movies, similarity

@st.cache_data
def fetch_poster(movie_id):
    try:
        response = requests.get(TMDB_URL.format(movie_id, API_KEY))
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"

def recommend(movie, movies, similarity):
    if movie not in movies['title'].values:
        st.error("Movie not found!")
        return [], []

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


# ---------------- APP START ---------------- #
st.title("🎬 Movie Recommendation System")

with st.spinner("Loading data and computing similarity..."):
    movies, similarity = load_data_and_compute_similarity()

selected_movie = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    names, posters = recommend(selected_movie, movies, similarity)
    if names:
        cols = st.columns(5)
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.image(poster)
                st.caption(name)
