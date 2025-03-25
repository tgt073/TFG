import pickle
import pandas as pd
import numpy as np
import warnings
import scipy.sparse as sp
from tqdm import tqdm
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, KFold
from sklearn.model_selection import train_test_split
import datetime

warnings.filterwarnings("ignore")

#########################################
# Funciones de Preprocesamiento         #
#########################################
def extract_year(date_str: str) -> int | None:
    """
    Extrae el año (como entero) a partir de una fecha en formato MM/DD/YYYY.
    Retorna None si la conversión falla.
    """
    try:
        return datetime.datetime.strptime(date_str, '%m/%d/%Y').year
    except Exception:
        return None

def load_preprocessed_data(ratings_filepath: str, movies_filepath: str):
    ratings_df = pd.read_csv(ratings_filepath, low_memory=False)
    movies_df = pd.read_csv(movies_filepath, low_memory=False)
    if 'imdbId' in ratings_df.columns:
        ratings_df.rename(columns={'imdbId': 'imdb_id'}, inplace=True)
    for df in [movies_df, ratings_df]:
        df['imdb_id'] = pd.to_numeric(df['imdb_id'], errors='coerce')
        df.dropna(subset=['imdb_id'], inplace=True)
        df['imdb_id'] = df['imdb_id'].astype(int)
    # Extraer el año a partir de "release_date" si existe
    if 'release_date' in movies_df.columns:
        movies_df['release_year'] = movies_df['release_date'].apply(lambda d: extract_year(d))
    return ratings_df, movies_df

#########################################
# Métodos para guardar/cargar modelos   #
#########################################
def save_model(algo, filename="svd_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(algo, f)

def load_model(filename="svd_model.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        return None

#########################################
# Implementación del Modelo SVD         #
#########################################
def train_svd(data):
    algo = SVD(n_factors=100, n_epochs=40, lr_all=0.01, reg_all=0.1)
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    return algo

def evaluate_with_kfold(algo, data, n_splits=5):
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    return cross_validate(algo, data, measures=['rmse', 'mae'], cv=kf, verbose=True)

#########################################
# Generación de Recomendaciones         #
#########################################
def get_svd_recommendations(user_id: int, movies_file: str, ratings_file: str, top_n: int = 10):
    ratings_df, movies_df = load_preprocessed_data(ratings_file, movies_file)
    train_ratings, _ = train_test_split(ratings_df, test_size=0.2, random_state=42)
    data = Dataset.load_from_df(train_ratings[['userId', 'imdb_id', 'rating']], Reader(rating_scale=(0, 5)))
    algo = load_model() or train_svd(data)
    if load_model() is None:
        save_model(algo)
    seen_movies = set(train_ratings.loc[train_ratings['userId'] == user_id, 'imdb_id'])
    unseen_movies = [m for m in movies_df['imdb_id'] if m not in seen_movies][:200]
    predictions = [algo.predict(user_id, mid) for mid in tqdm(unseen_movies)]
    pred_df = pd.DataFrame([(int(pred.iid), np.clip(pred.est, 0, 5)) for pred in predictions],
                           columns=['imdb_id', 'predicted_rating'])
    movies_df = movies_df.drop_duplicates(subset=['imdb_id'])
    # Incluir 'release_year' y 'poster_path' en el merge para disponer de esa información
    pred_df = pred_df.merge(
        movies_df[['imdb_id', 'title', 'release_year', 'imdb_votes', 'imdb_rating', 'poster_path']],
        on='imdb_id', how='left'
    )
    return pred_df.sort_values(by='predicted_rating', ascending=False).head(top_n).to_dict(orient='records')
