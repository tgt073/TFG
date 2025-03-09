import pandas as pd
import numpy as np
import math
import warnings
import pickle
import re
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from annoy import AnnoyIndex
import scipy.sparse as sp

warnings.filterwarnings("ignore")


#########################################
# Funciones de Preprocesamiento         #
#########################################
def load_preprocessed_data(ratings_filepath: str, movies_filepath: str):
	ratings_df = pd.read_csv(ratings_filepath, low_memory=False)
	movies_df = pd.read_csv(movies_filepath, low_memory=False)

	if 'imdbId' in ratings_df.columns:
		ratings_df.rename(columns={'imdbId': 'imdb_id'}, inplace=True)

	for df in [movies_df, ratings_df]:
		df['imdb_id'] = df['imdb_id'].apply(lambda x: int(re.sub(r'\D', '', str(x))) if pd.notnull(x) else x)
		df.dropna(subset=['imdb_id'], inplace=True)
		df['imdb_id'] = df['imdb_id'].astype(int)

	return ratings_df, movies_df


def filter_ratings(ratings_df: pd.DataFrame, min_user_ratings: int = 5, min_item_ratings: int = 5) -> pd.DataFrame:
	user_counts = ratings_df.groupby('userId').size()
	valid_users = user_counts[user_counts >= min_user_ratings].index
	ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
	item_counts = ratings_df.groupby('imdb_id').size()
	valid_items = item_counts[item_counts >= min_item_ratings].index
	return ratings_df[ratings_df['imdb_id'].isin(valid_items)]


def create_user_item_matrix(ratings_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	user_item = ratings_df.pivot(index='userId', columns='imdb_id', values='rating')
	user_means = user_item.mean(axis=1)
	user_stds = user_item.std(axis=1).replace(0, 1)
	normalized = (user_item.sub(user_means, axis=0)).div(user_stds, axis=0).fillna(0)
	return normalized, user_means


#########################################
# Precomputación de vecinos con Annoy
#########################################
def precompute_topk_with_annoy(user_item: pd.DataFrame, k: int, n_components: int = 50, n_trees: int = 10) -> tuple[
	dict, TruncatedSVD]:
	"""
	Reduce la matriz usuario‑item a n_components usando SVD y construye un índice Annoy (angular).
	Luego, para cada usuario se obtienen sus k vecinos más similares (excluyendo a sí mismo).
	Retorna un diccionario { usuario: (lista_vecinos, lista_similitudes) } y el modelo SVD.
	"""
	# Reducir dimensionalidad para acelerar la búsqueda
	svd = TruncatedSVD(n_components=n_components, random_state=42)
	reduced = svd.fit_transform(user_item.values)  # shape: (n_users, n_components)
	d = n_components

	# Construir el índice Annoy
	index = AnnoyIndex(d, 'angular')
	user_ids = user_item.index.tolist()
	for i, vector in enumerate(reduced):
		index.add_item(i, vector)
	index.build(n_trees)

	# Para cada usuario, obtener sus k vecinos (excluyendo al propio usuario)
	topk_sim = {}
	for i, uid in enumerate(user_ids):
		indices = index.get_nns_by_item(i, k + 1)  # El primero es el propio usuario
		neighbors = [user_ids[j] for j in indices if j != i][:k]
		sims = []
		vec = reduced[i]
		for j in indices:
			if j == i:
				continue
			other_vec = reduced[j]
			sim = np.dot(vec, other_vec) / (np.linalg.norm(vec) * np.linalg.norm(other_vec))
			sims.append(sim)
		topk_sim[uid] = (neighbors, sims)
	return topk_sim, svd


#########################################
# Implementación del Modelo PearsonCF   #
#########################################
class PearsonCF:
	def __init__(self, ratings_df: pd.DataFrame, k: int = 20, m_threshold: int = 200, n_components: int = 50,
	             n_trees: int = 10):
		self.ratings_df = ratings_df
		self.k = k
		self.m_threshold = m_threshold
		self.n_components = n_components
		self.n_trees = n_trees

		# Cargar o calcular la matriz usuario-item normalizada y las medias
		try:
			with open("user_item_matrix.pkl", "rb") as f:
				self.user_item = pickle.load(f)
			with open("user_means.pkl", "rb") as f:
				self.user_means = pickle.load(f)
			print("✅ Matriz de usuario-items y medias cargadas desde caché.")
		except FileNotFoundError:
			self.user_item, self.user_means = create_user_item_matrix(ratings_df)
			with open("user_item_matrix.pkl", "wb") as f:
				pickle.dump(self.user_item, f)
			with open("user_means.pkl", "wb") as f:
				pickle.dump(self.user_means, f)
			print("✅ Matriz de usuario-items y medias calculadas y guardadas.")

		# Precomputar vecinos top-k usando Annoy y reducción de dimensionalidad
		try:
			with open("topk_annoy.pkl", "rb") as f:
				self.topk_sim, self.svd_model = pickle.load(f)
			print("✅ Top-k vecinos (Annoy) cargados desde caché.")
		except FileNotFoundError:
			print("⏳ Precomputando top-k vecinos con Annoy...")
			self.topk_sim, self.svd_model = precompute_topk_with_annoy(self.user_item, self.k, self.n_components,
			                                                           self.n_trees)
			with open("topk_annoy.pkl", "wb") as f:
				pickle.dump((self.topk_sim, self.svd_model), f)
			print("✅ Top-k vecinos (Annoy) precomputados y guardados.")

	def predict_rating(self, user: int, movie: int) -> float:
		if movie not in self.user_item.columns or user not in self.user_item.index:
			return np.nan

		valid_users = set(self.user_item[self.user_item[movie] != 0].index)
		if not valid_users:
			return np.nan

		top_users, top_sims = self.topk_sim.get(user, ([], []))
		if not top_users:
			return self.user_means[user]

		filtered = [(u, s) for u, s in zip(top_users, top_sims) if u in valid_users]
		if not filtered:
			return self.user_means[user]
		filtered_users, filtered_sims = zip(*filtered)
		filtered_users = np.array(filtered_users)
		filtered_sims = np.array(filtered_sims)

		if np.sum(np.abs(filtered_sims)) == 0:
			return self.user_means[user]

		ratings_top = self.user_item.loc[filtered_users, movie]
		weighted_sum = np.dot(ratings_top, filtered_sims)
		predicted = weighted_sum / np.sum(np.abs(filtered_sims)) + self.user_means[user]
		return np.clip(predicted, 0, 5)

	def recommend_movies(self, user: int, movies_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
		rated_movies = self.user_item.loc[user][self.user_item.loc[user] != 0].index
		unseen_movies = list(set(movies_df['imdb_id']) - set(rated_movies))
		predictions = [self.predict_rating(user, movie) for movie in unseen_movies]
		pred_df = pd.DataFrame({'imdb_id': unseen_movies, 'predicted_rating': predictions})
		pred_df.dropna(inplace=True)
		pred_df = pred_df.merge(movies_df[['imdb_id', 'title', 'imdb_votes', 'imdb_rating']], on='imdb_id', how='left')
		pred_df = pred_df[pred_df['imdb_votes'] >= self.m_threshold]
		return pred_df.sort_values(by='predicted_rating', ascending=False).head(top_n)


#########################################
# Interfaz para la aplicación Flask     #
#########################################
def get_user_based_recommendations(user_id: int, movies_file: str, ratings_file: str, top_n: int = 10):
	ratings_df, movies_df = load_preprocessed_data(ratings_file, movies_file)
	ratings_df = filter_ratings(ratings_df, min_user_ratings=20, min_item_ratings=20)
	train_ratings, _ = train_test_split(ratings_df, test_size=0.2, random_state=42)
	model = PearsonCF(train_ratings)
	recommendations = model.recommend_movies(user_id, movies_df, top_n)

	# Formatear la salida en tabla
	output_str = f"🔮 Recomendaciones para el usuario {user_id}:\n"
	header = "{:<40} {:<20} {:<15} {:<15}".format("Título", "Predicted Rating", "imdb_rating", "imdb_votes")
	output_str += header + "\n"
	output_str += "-" * len(header) + "\n"

	for rec in recommendations.to_dict(orient='records'):
		output_str += "{:<40} {:<20.2f} {:<15.2f} {:<15}\n".format(
			rec['title'][:40],
			rec['predicted_rating'],
			rec['imdb_rating'],
			int(rec['imdb_votes'])
		)

	print(output_str)
	return recommendations.to_dict(orient='records')
