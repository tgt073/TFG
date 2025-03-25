import pandas as pd
import numpy as np
import math
import warnings
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import re
from sklearn.metrics.pairwise import cosine_similarity
import datetime

warnings.filterwarnings("ignore")

def extract_year(date_str: str) -> int | None:
    """
    Extrae el año (como entero) a partir de una fecha en formato MM/DD/YYYY.
    Retorna None si la conversión falla.
    """
    try:
        return datetime.datetime.strptime(date_str, '%m/%d/%Y').year
    except Exception:
        return None

#########################################
# Funciones de Preprocesamiento         #
#########################################
def load_preprocessed_data(ratings_filepath: str, movies_filepath: str):
    ratings_df = pd.read_csv(ratings_filepath, low_memory=False)
    movies_df = pd.read_csv(movies_filepath, low_memory=False)
    for df in [movies_df, ratings_df]:
        if 'imdbId' in df.columns:
            df.rename(columns={'imdbId': 'imdb_id'}, inplace=True)
        df['imdb_id'] = df['imdb_id'].apply(
            lambda x: re.search(r'\d+', str(x)).group() if re.search(r'\d+', str(x)) else x)
        df['imdb_id'] = pd.to_numeric(df['imdb_id'], errors='coerce')
        df.dropna(subset=['imdb_id'], inplace=True)
        df['imdb_id'] = df['imdb_id'].astype(int)
    movies_df.drop_duplicates(subset=['imdb_id'], inplace=True)
    # Extraemos el año a partir del campo "release_date"
    if 'release_date' in movies_df.columns:
        movies_df['release_year'] = movies_df['release_date'].apply(lambda d: extract_year(d))
    # Filtramos la película problemática: por ejemplo, eliminamos "Gladiator" de 1992
    movies_df = movies_df[~((movies_df['title'].str.lower() == 'gladiator') &
                              (movies_df['release_year'] == 1992))]
    movies_df = movies_df[~((movies_df['title'].str.lower() == 'the bourne identity') &
                            (movies_df['release_year'] == 1988))]
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
# Implementación del Modelo ItemBasedCF #
#########################################
class ItemBasedCF:
    def __init__(self, ratings_df: pd.DataFrame, k: int = 20, m_threshold: int = 150, lambda_shrink: float = 10):
        self.ratings_df = ratings_df
        self.user_item = create_user_item_matrix(ratings_df)
        self.k = k
        self.m_threshold = m_threshold
        self.lambda_shrink = lambda_shrink
        self.item_similarity = self.load_similarity_matrix()
        if self.item_similarity is None or set(self.item_similarity.index) != set(self.user_item.columns):
            self.compute_similarity()
            self.save_similarity_matrix()

    def load_similarity_matrix(self, filename="item_similarity.pkl"):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError):
            return None

    def save_similarity_matrix(self, filename="item_similarity.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.item_similarity, f)

    def compute_similarity(self):
        similarity = cosine_similarity(self.user_item.T)
        similarity = np.nan_to_num(similarity) / (1 + self.lambda_shrink)
        self.item_similarity = pd.DataFrame(similarity, index=self.user_item.columns, columns=self.user_item.columns)

    def predict_rating(self, user: int, movie: int) -> float:
        if movie not in self.user_item.columns or user not in self.user_item.index:
            return np.nan
        user_ratings = self.user_item.loc[user]
        rated_movies = user_ratings[user_ratings > 0].index
        sim_scores = self.item_similarity.loc[movie, rated_movies]
        top_k = sim_scores.abs().nlargest(self.k)
        if top_k.sum() == 0:
            return np.nan
        return np.dot(user_ratings[top_k.index], top_k) / top_k.sum()

    def recommend_movies(self, user: int, movies_df: pd.DataFrame, top_n: int = 10) -> list:
        rated_movies = self.user_item.loc[user][self.user_item.loc[user] > 0].index
        unseen_movies = list(set(movies_df['imdb_id']) - set(rated_movies))
        predictions = (self.predict_rating(user, movie) for movie in unseen_movies)
        pred_df = pd.DataFrame({'imdb_id': unseen_movies, 'predicted_rating': predictions}).dropna()
        pred_df = pred_df.merge(
            movies_df[['imdb_id', 'title', 'release_year', 'imdb_votes', 'imdb_rating', 'poster_path']],
            on='imdb_id', how='left'
        )
        pred_df = pred_df[(pred_df['imdb_rating'] > 5) & (pred_df['imdb_votes'] >= self.m_threshold)]
        return pred_df.sort_values(by='predicted_rating', ascending=False).head(top_n).to_dict(orient='records')

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
    # Rutas a los archivos (ajusta las rutas según tu entorno)
    ratings_filepath = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_ratings.csv'
    movies_filepath = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_movies.csv'

    # Definimos un usuario de prueba (asegúrate de que exista en el dataset de ratings)
    test_user_id = 1

    # Obtenemos las recomendaciones basadas en ítems para el usuario de prueba
    recommendations = get_item_based_recommendations(test_user_id, movies_filepath, ratings_filepath, top_n=10)

    # Imprimimos las recomendaciones
    print(f"Recomendaciones para el usuario {test_user_id}:")
    for rec in recommendations:
        print("Título:", rec.get('title'))
        print("Año:", rec.get('release_year'))
        print("Puntuación IMDb:", rec.get('imdb_rating'))
        print("Votos IMDb:", rec.get('imdb_votes'))
        print("Poster:", rec.get('poster_path'))
        print("-" * 40)
