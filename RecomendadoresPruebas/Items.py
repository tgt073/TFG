import pandas as pd
import numpy as np
import math
import warnings
import pickle
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import re
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

#########################################
# Funciones de Preprocesamiento         #
#########################################
def load_preprocessed_data(ratings_filepath: str, movies_filepath: str):
    ratings_df = pd.read_csv(ratings_filepath, low_memory=False)
    movies_df = pd.read_csv(movies_filepath, low_memory=False)

    # Si la columna viene como "imdbId", renombrarla a "imdb_id"
    if 'imdbId' in ratings_df.columns:
        ratings_df.rename(columns={'imdbId': 'imdb_id'}, inplace=True)
    if 'imdbId' in movies_df.columns:
        movies_df.rename(columns={'imdbId': 'imdb_id'}, inplace=True)

    # Convertir la columna "imdb_id" a numérico extrayendo dígitos (ej. "tt1234567")
    for df in [movies_df, ratings_df]:
        if df['imdb_id'].dtype == object:
            df['imdb_id'] = df['imdb_id'].apply(
                lambda x: re.search(r'\d+', str(x)).group() if re.search(r'\d+', str(x)) else x)
        df['imdb_id'] = pd.to_numeric(df['imdb_id'], errors='coerce')
        df.dropna(subset=['imdb_id'], inplace=True)
        df['imdb_id'] = df['imdb_id'].astype(int)

    # Eliminar películas duplicadas en el dataset de películas
    movies_df.drop_duplicates(subset=['imdb_id'], inplace=True)

    return ratings_df, movies_df

def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    user_item = ratings_df.pivot(index='userId', columns='imdb_id', values='rating').fillna(0)
    return user_item

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

        # Intentar cargar la matriz de similitud; si no existe o no coincide con el dataset actual, se recalcula.
        try:
            with open("item_similarity.pkl", "rb") as f:
                self.item_similarity = pickle.load(f)
            if set(self.item_similarity.index) != set(self.user_item.columns):
                print("Dimensiones de la similitud cargada no coinciden. Recalculando...")
                self.compute_similarity()
        except FileNotFoundError:
            self.compute_similarity()

    def compute_similarity(self):
        # Usar cosine_similarity, que está optimizada para grandes matrices
        similarity = cosine_similarity(self.user_item.T)
        similarity = np.nan_to_num(similarity)
        similarity /= (1 + self.lambda_shrink)
        self.item_similarity = pd.DataFrame(similarity, index=self.user_item.columns, columns=self.user_item.columns)
        with open("item_similarity.pkl", "wb") as f:
            pickle.dump(self.item_similarity, f)

    def predict_rating(self, user: int, movie: int) -> float:
        if movie not in self.user_item.columns or user not in self.user_item.index:
            return np.nan

        user_ratings = self.user_item.loc[user]
        rated_movies = user_ratings[user_ratings > 0].index
        if len(rated_movies) == 0:
            return np.nan

        sim_scores = self.item_similarity.loc[movie, rated_movies]
        top_k = sim_scores.abs().nlargest(self.k)
        top_movies = top_k.index
        top_sim = sim_scores.loc[top_movies]
        top_ratings = user_ratings.loc[top_movies]

        if top_sim.sum() == 0:
            return np.nan

        pred = np.dot(top_ratings, top_sim) / top_sim.sum()
        return np.clip(pred, 0, 5)

    def recommend_movies(self, user: int, movies_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        rated_movies = self.user_item.loc[user][self.user_item.loc[user] > 0].index
        unseen_movies = list(set(movies_df['imdb_id']) - set(rated_movies))
        predictions = Parallel(n_jobs=-1)(delayed(self.predict_rating)(user, movie) for movie in unseen_movies)
        pred_df = pd.DataFrame({'imdb_id': unseen_movies, 'predicted_rating': predictions})
        pred_df.dropna(inplace=True)

        pred_df = pred_df.merge(movies_df[['imdb_id', 'title', 'imdb_votes', 'imdb_rating']], on='imdb_id', how='left')
        # Filtrar para que solo se incluyan películas con calificación mayor a 5 y votos suficientes
        pred_df = pred_df[pred_df['imdb_rating'] > 5]
        pred_df = pred_df[pred_df['imdb_votes'] >= self.m_threshold]
        return pred_df.sort_values(by='predicted_rating', ascending=False).head(top_n)




#########################################
# Evaluación del Modelo                 #
#########################################
def evaluate_with_kfold(model, ratings_df, n_splits=5):
	kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
	actual, predicted = [], []

	for train_index, test_index in kf.split(ratings_df):
		train_ratings, test_ratings = ratings_df.iloc[train_index], ratings_df.iloc[test_index]
		for _, row in test_ratings.iterrows():
			pred_rating = model.predict_rating(row['userId'], row['imdb_id'])
			if not np.isnan(pred_rating):
				actual.append(row['rating'])
				predicted.append(pred_rating)
	if len(actual) == 0:
		print("❌ No hay predicciones válidas. No se puede evaluar.")
		return None
	actual, predicted = np.array(actual), np.array(predicted)
	rmse = math.sqrt(mean_squared_error(actual, predicted))
	mae = np.mean(np.abs(actual - predicted))

	print("\n📊 Evaluación del modelo PearsonCF:")
	print(f"✅ RMSE: {rmse:.4f}")
	print(f"✅ MAE:  {mae:.4f}")
	return rmse, mae

#########################################
# Interfaz para la aplicación web       #
#########################################
def get_item_based_recommendations(user_id: int, movies_file: str, ratings_file: str, top_n: int = 10):
    ratings_df, movies_df = load_preprocessed_data(ratings_file, movies_file)
    ratings_df = filter_ratings(ratings_df, min_user_ratings=20, min_item_ratings=20)
    # Entrenamos el modelo con el conjunto de ratings filtrados
    train_ratings, _ = train_test_split(ratings_df, test_size=0.2, random_state=42)
    model = ItemBasedCF(train_ratings)
    # evaluate_with_kfold(model, _)  # La evaluación se puede omitir en producción
    recommendations = model.recommend_movies(user_id, movies_df, top_n)
    return recommendations.to_dict(orient='records')
