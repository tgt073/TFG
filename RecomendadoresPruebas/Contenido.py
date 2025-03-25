import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize


#########################################
# Funciones Auxiliares
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


#########################################
# Funciones de Preprocesamiento         #
#########################################
def load_csv(filepath: str) -> pd.DataFrame:
	return pd.read_csv(filepath, engine='python', on_bad_lines='skip')


def preprocess_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
	"""
    Preprocesa el DataFrame de películas adaptado a las nuevas columnas.
    """
	movies_df = movies_df.copy()  # Evita advertencias de asignación en copia

	# Rellenar valores nulos con cadenas vacías
	for col in ['overview', 'tagline', 'genres', 'keywords', 'director', 'main_actors']:
		movies_df[col] = movies_df[col].fillna('')

	# Convertir imdb_votes a numérico
	movies_df['imdb_votes'] = pd.to_numeric(movies_df['imdb_votes'], errors='coerce').fillna(0)

	# Filtrar películas con suficientes votos
	movies_df = movies_df[movies_df['imdb_votes'] >= 1000]

	# Normalizar los títulos y eliminar duplicados
	movies_df.loc[:, 'title'] = movies_df['title'].str.strip().str.lower()
	movies_df = movies_df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)

	# Si existe "release_date", creamos la columna "release_year" extrayendo el año
	if 'release_date' in movies_df.columns:
		movies_df['release_year'] = movies_df['release_date'].apply(lambda d: extract_year(d))

	return movies_df


def save_vectorizer(vectorizer, filename):
	with open(filename, "wb") as f:
		pickle.dump(vectorizer, f)


def load_vectorizer(filename):
	try:
		with open(filename, "rb") as f:
			return pickle.load(f)
	except (FileNotFoundError, pickle.UnpicklingError):
		return None


#########################################
# Sistema de Recomendación basado en contenido
#########################################
def get_content_based_recommendations(movie_title: str, movies_file: str, num_recommendations: int = 10,
                                      weights: dict = None):
	"""
	Obtiene recomendaciones basadas en contenido.
	Parámetros:
	  movie_title: título de la película a consultar.
	  movies_file: ruta al CSV de películas.
	  num_recommendations: número de películas recomendadas a devolver.
	  weights: diccionario con pesos para combinar las similitudes.
	"""
	movies_df = load_csv(movies_file)
	movies_df = preprocess_movies(movies_df)
	title = movie_title.lower()
	indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
	if title not in indices:
		return []
	idx = indices[title]

	# Inicializar los vectorizadores para cada campo
	vectorizers = {
		'overview': TfidfVectorizer(stop_words='english'),
		'tagline': TfidfVectorizer(stop_words='english'),
		'genres': CountVectorizer(stop_words='english'),
		'keywords': CountVectorizer(stop_words='english'),
		'director': CountVectorizer(stop_words='english'),
		'main_actors': CountVectorizer(stop_words='english')
	}

	matrices = {}
	for key, vectorizer in vectorizers.items():
		cached_vectorizer = load_vectorizer(f"{key}_vectorizer.pkl")
		if cached_vectorizer:
			vectorizer = cached_vectorizer
		else:
			save_vectorizer(vectorizer, f"{key}_vectorizer.pkl")
		matrices[key] = vectorizer.fit_transform(movies_df[key])
		if key in ['genres', 'keywords', 'director', 'main_actors']:
			matrices[key] = normalize(matrices[key], norm='l2')

	# Usar los pesos pasados o los valores por defecto
	if weights is None:
		weights = {
			'overview': 0.25,
			'tagline': 0.05,
			'genres': 0.25,
			'keywords': 0.05,
			'director': 0.15,
			'main_actors': 0.25
		}

	# Calcular la similitud combinada usando los pesos
	combined_scores = np.zeros(movies_df.shape[0])
	for name, matrix in matrices.items():
		sim = linear_kernel(matrix[idx], matrix)
		combined_scores += weights.get(name, 0) * sim.ravel()

	# Obtener los índices de las películas más similares (excluyendo la misma película)
	sim_indices = combined_scores.argsort()[::-1][1:num_recommendations + 1]
	results_df = movies_df.loc[
		sim_indices, ['title', 'release_year', 'imdb_votes', 'imdb_rating', 'poster_path']].copy()
	results_df = results_df[results_df['imdb_rating'] > 5]
	results_df.reset_index(drop=True, inplace=True)

	# Convertir los títulos a mayúsculas y minúsculas (title case)
	results_df['title'] = results_df['title'].str.title()

	return results_df.to_dict(orient='records')
