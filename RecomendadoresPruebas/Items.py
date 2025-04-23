import os
import re
import datetime
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

#########################################
# Utilidades para la caché              #
#########################################
def ensure_cache_dir(subdir="cache"):
    base_path = os.path.dirname(os.path.abspath(__file__))  # Este archivo está en RecomendadoresPruebas
    cache_path = os.path.join(base_path, subdir)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    return cache_path

def get_cache_path(filename, subdir="cache"):
    return os.path.join(ensure_cache_dir(subdir), filename)

#########################################
# Funciones de Preprocesamiento         #
#########################################
def extract_year(date_str: str) -> int | None:
    try:
        return datetime.datetime.strptime(date_str, '%m/%d/%Y').year
    except Exception:
        return None

def load_preprocessed_data(ratings_filepath: str, movies_filepath: str):
    ratings_df = pd.read_csv(ratings_filepath, low_memory=False)
    movies_df = pd.read_csv(movies_filepath, low_memory=False)

    for df in [ratings_df, movies_df]:
        if 'imdbId' in df.columns:
            df.rename(columns={'imdbId': 'imdb_id'}, inplace=True)
        df['imdb_id'] = df['imdb_id'].apply(
            lambda x: re.search(r'\d+', str(x)).group() if re.search(r'\d+', str(x)) else x)
        df['imdb_id'] = pd.to_numeric(df['imdb_id'], errors='coerce')
        df.dropna(subset=['imdb_id'], inplace=True)
        df['imdb_id'] = df['imdb_id'].astype(int)

    movies_df.drop_duplicates(subset=['imdb_id'], inplace=True)

    if 'release_date' in movies_df.columns:
        movies_df['release_year'] = movies_df['release_date'].apply(lambda d: extract_year(d))

    # Eliminamos películas problemáticas
    movies_df = movies_df[~((movies_df['title'].str.lower() == 'gladiator') &
                            (movies_df.get('release_year') == 1992))]
    movies_df = movies_df[~((movies_df['title'].str.lower() == 'the bourne identity') &
                            (movies_df.get('release_year') == 1988))]

    return ratings_df, movies_df

def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    return ratings_df.pivot(index='userId', columns='imdb_id', values='rating').fillna(0)

def filter_ratings(ratings_df: pd.DataFrame, min_user_ratings: int = 5, min_item_ratings: int = 5) -> pd.DataFrame:
    user_counts = ratings_df.groupby('userId').size()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
    item_counts = ratings_df.groupby('imdb_id').size()
    valid_items = item_counts[item_counts >= min_item_ratings].index
    return ratings_df[ratings_df['imdb_id'].isin(valid_items)]

#########################################
# Modelo ItemBasedCF con Similitud Ajustada
#########################################
class ItemBasedCF:
    def __init__(self, ratings_df: pd.DataFrame, k: int = 20, m_threshold: int = 150, lambda_shrink: float = 10):
        self.ratings_df = ratings_df
        self.user_item = create_user_item_matrix(ratings_df)
        self.k = k
        self.m_threshold = m_threshold
        self.lambda_shrink = lambda_shrink

        # Intentar cargar la matriz de similitud desde caché
        try:
            with open(get_cache_path("item_similarity.pkl"), "rb") as f:
                self.item_similarity = pickle.load(f)
        except FileNotFoundError:
            self.compute_similarity()

    def compute_similarity(self):
        centered_matrix = self.user_item.sub(self.user_item.mean(axis=1), axis=0)
        sim = cosine_similarity(centered_matrix.T)
        sim = np.nan_to_num(sim) / (1 + self.lambda_shrink)
        self.item_similarity = pd.DataFrame(sim, index=self.user_item.columns, columns=self.user_item.columns)

        with open(get_cache_path("item_similarity.pkl"), "wb") as f:
            pickle.dump(self.item_similarity, f)

    def predict_rating(self, user: int, movie: int) -> float:
        movie = int(movie)
        if movie not in self.user_item.columns or user not in self.user_item.index:
            return np.nan

        user_ratings = self.user_item.loc[user]
        if (user_ratings > 0).sum() == 0:
            return np.nan

        user_mean = user_ratings[user_ratings > 0].mean()
        rated_movies = user_ratings[user_ratings > 0].index.astype(int)
        rated_movies = [m for m in rated_movies if m in self.item_similarity.index]
        if not rated_movies:
            return np.nan

        centered_ratings = user_ratings.loc[rated_movies] - user_mean
        sim_scores = self.item_similarity.loc[movie, rated_movies]
        top_k = sim_scores.abs().nlargest(self.k)
        if top_k.sum() == 0:
            return np.nan

        weighted_sum = np.dot(centered_ratings.loc[top_k.index], top_k)
        pred = user_mean + (weighted_sum / top_k.sum())
        return np.clip(pred, 0, 5)

    def recommend_movies(self, user: int, movies_df: pd.DataFrame, top_n: int = 10) -> list:
        print(f"Intentando recomendar películas para el usuario: {user}")

        if user not in self.user_item.index:
            print("El usuario no está en la matriz de usuario-ítem después del filtrado.")
            return []

        rated_movies = self.user_item.loc[user][self.user_item.loc[user] > 0].index
        unseen_movies = set(self.user_item.columns) - set(rated_movies)
        print(f"Películas calificadas: {len(rated_movies)}, Películas no vistas: {len(unseen_movies)}")

        predicciones = []
        for movie in unseen_movies:
            pred = self.predict_rating(user, movie)
            if not np.isnan(pred):
                predicciones.append((movie, pred))

        if not predicciones:
            print("No se generaron predicciones válidas.")
            return []

        pred_df = pd.DataFrame(predicciones, columns=['imdb_id', 'predicted_rating'])
        pred_df = pred_df.merge(movies_df[['imdb_id', 'title', 'imdb_rating', 'imdb_votes']], on='imdb_id', how='left')

        print(f"Predicciones totales antes del filtrado por votos: {len(pred_df)}")
        pred_df = pred_df[(pred_df['imdb_rating'] > 5) & (pred_df['imdb_votes'] >= self.m_threshold)]
        print(f"Predicciones tras filtrado por votos: {len(pred_df)}")

        return pred_df.sort_values('predicted_rating', ascending=False).head(top_n).to_dict(orient='records')

#########################################
# Interfaz para la aplicación web       #
#########################################
def get_item_based_recommendations(user_id: int, movies_file: str, ratings_file: str, top_n: int = 10):
    ratings_df, movies_df = load_preprocessed_data(ratings_file, movies_file)
    ratings_df = filter_ratings(ratings_df, min_user_ratings=20, min_item_ratings=20)
    train_ratings, _ = train_test_split(ratings_df, test_size=0.2, random_state=42)
    model = ItemBasedCF(train_ratings)
    return model.recommend_movies(user_id, movies_df, top_n)

if __name__ == '__main__':
    ratings_filepath = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_ratings.csv'
    movies_filepath = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_movies.csv'
    test_user_id = 1

    recommendations = get_item_based_recommendations(test_user_id, movies_filepath, ratings_filepath, top_n=10)

    print(f"Recomendaciones para el usuario {test_user_id}:")
    for rec in recommendations:
        print("Título:", rec.get('title'))
        print("Año:", rec.get('release_year'))
        print("Puntuación IMDb:", rec.get('imdb_rating'))
        print("Votos IMDb:", rec.get('imdb_votes'))
        print("Poster:", rec.get('poster_path'))
        print("-" * 40)
