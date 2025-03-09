import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize


#########################################
# Funciones de Preprocesamiento         #
#########################################
def load_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, low_memory=False)


def preprocess_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa el DataFrame de películas adaptado a las nuevas columnas.
    """
    # Rellenar valores nulos con cadenas vacías
    movies_df['overview'] = movies_df['overview'].fillna('')
    movies_df['tagline'] = movies_df['tagline'].fillna('')
    movies_df['genres'] = movies_df['genres'].fillna('')
    movies_df['keywords'] = movies_df['keywords'].fillna('')
    movies_df['director'] = movies_df['director'].fillna('')
    movies_df['main_actors'] = movies_df['main_actors'].fillna('')

    # Filtrar películas con suficientes votos
    if 'imdb_votes' in movies_df.columns:
        movies_df = movies_df[movies_df['imdb_votes'] >= 1000]

    # Normalizar los títulos de las películas
    movies_df['title'] = movies_df['title'].str.strip().str.lower()
    movies_df.drop_duplicates(subset=['title'], keep='first', inplace=True)
    movies_df.reset_index(drop=True, inplace=True)
    return movies_df


#########################################
# Sistema de Recomendación basado en contenido
#########################################
def get_content_based_recommendations(movie_title: str, movies_file: str, num_recommendations: int = 10):
    movies_df = load_csv(movies_file)
    movies_df = preprocess_movies(movies_df)

    # Encontrar la película por título
    title = movie_title.lower()
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    if title not in indices:
        return []
    idx = indices[title]

    # Vectorización de características
    tfidf_overview = TfidfVectorizer(stop_words='english')
    tfidf_tagline = TfidfVectorizer(stop_words='english')
    count_genres = CountVectorizer(stop_words='english')
    count_keywords = CountVectorizer(stop_words='english')
    count_director = CountVectorizer(stop_words='english')
    count_actors = CountVectorizer(stop_words='english')

    overview_matrix = tfidf_overview.fit_transform(movies_df['overview'])
    tagline_matrix = tfidf_tagline.fit_transform(movies_df['tagline'])
    genres_matrix = count_genres.fit_transform(movies_df['genres'])
    keywords_matrix = count_keywords.fit_transform(movies_df['keywords'])
    director_matrix = count_director.fit_transform(movies_df['director'])
    actors_matrix = count_actors.fit_transform(movies_df['main_actors'])

    # Normalización de los vectores
    genres_matrix = normalize(genres_matrix, norm='l2')
    keywords_matrix = normalize(keywords_matrix, norm='l2')
    director_matrix = normalize(director_matrix, norm='l2')
    actors_matrix = normalize(actors_matrix, norm='l2')

    # Pesos de cada característica
    weights = {
        'overview': 0.3,
        'tagline': 0.05,
        'genres': 0.25,
        'keywords': 0.05,
        'director': 0.2,
        'actors': 0.15
    }

    # Cálculo de similitudes
    combined_scores = np.zeros(movies_df.shape[0])
    for name, matrix in zip(weights.keys(),
                            [overview_matrix, tagline_matrix, genres_matrix, keywords_matrix, director_matrix,
                             actors_matrix]):
        sim = linear_kernel(matrix[idx], matrix)
        combined_scores += weights[name] * sim.ravel()

    # Obtener las recomendaciones
    sim_indices = combined_scores.argsort()[::-1][1:num_recommendations + 1]
    results_df = movies_df.loc[sim_indices, ['title', 'imdb_votes', 'imdb_rating']].copy()

    # Filtrar películas con imdb_rating mayor a 5
    results_df = results_df[results_df['imdb_rating'] > 5]

    results_df.reset_index(drop=True, inplace=True)
    return results_df.to_dict(orient='records')
