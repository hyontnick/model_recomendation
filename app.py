import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from surprise import SVD

# Charger les fichiers sauvegardés
with open('best_svd_model.pkl', 'rb') as model_file:
    best_svd = pickle.load(model_file)

with open('genre_means.pkl', 'rb') as genre_file:
    genre_means = pickle.load(genre_file)

# Charger les données (ici, on utilise un échantillon pour la démonstration)
ratings = pd.read_csv('ml-25m/ratings.csv').head(60000)
movies = pd.read_csv('ml-25m/movies.csv').head(60000)
data = pd.merge(ratings, movies, on='movieId')
le = LabelEncoder()
data['genres_encoded'] = le.fit_transform(data['genres'])

# Normaliser les colonnes 'userId' et 'genres_encoded'
scaler = MinMaxScaler()
data[['userId', 'genres_encoded']] = scaler.fit_transform(data[['userId', 'genres_encoded']])

# Fonction pour faire une prédiction basée sur le contenu
def content_based_prediction(user_id, genres_encoded):
    return genre_means.get(genres_encoded, 3.0)  # Retourner une note moyenne ou 3.0 par défaut

# Fonction pour faire des prédictions hybrides
def hybrid_prediction(user_id, genres_encoded):
    svd_pred = best_svd.predict(user_id, genres_encoded).est
    content_pred = content_based_prediction(user_id, genres_encoded)
    hybrid_pred = (svd_pred + content_pred) / 2
    return hybrid_pred

# Interface Streamlit
st.title('Système de Recommandation Hybride')

user_id = st.number_input('Entrez l\'ID de l\'utilisateur:', min_value=1)
movie_id = st.number_input('Entrez l\'ID du film:', min_value=1)

if st.button('Faire une prédiction'):
    if not data[data['movieId'] == movie_id].empty:
        genres_encoded = data[data['movieId'] == movie_id]['genres_encoded'].values[0]
        prediction = hybrid_prediction(user_id, genres_encoded)
        st.write(f"La prédiction hybride pour l'utilisateur {user_id} et le film {movie_id} est {prediction:.2f}")
    else:
        st.write(f"Le film avec l'ID {movie_id} n'existe pas dans les données.")

