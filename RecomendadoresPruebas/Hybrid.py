import os
import pandas as pd
from RecomendadoresPruebas.Contenido import get_content_based_recommendations
from RecomendadoresPruebas.Modelo import get_svd_recommendations


def normalize_scores(recs, key):
    """Normaliza las puntuaciones de un listado de recomendaciones."""
    scores = [rec.get(key, 0) for rec in recs]
    if not scores:
        return recs
    min_val, max_val = min(scores), max(scores)
    for rec in recs:
        rec[key + '_norm'] = (rec.get(key, 0) - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    return recs


def ensure_cache_dir(cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def get_hybrid_recommendations_cascade(user_id, movie_title, movies_file, ratings_file, top_n=20, candidate_n=1000):
    """
    Enfoque en cascada:
      1. Se generan candidatos mediante el recomendador basado en contenido.
      2. Se obtienen las predicciones SVD (para un amplio conjunto) y se filtran
         aquellas cuyo título (normalizado a minúsculas) esté en el conjunto de candidatos.
      3. Se ordenan por 'predicted_rating' y se devuelven las top_n recomendaciones.

    Devuelve una lista de diccionarios.
    """
    # Paso 1: Generar candidatos basados en contenido
    rec_content = get_content_based_recommendations(movie_title, movies_file, num_recommendations=candidate_n)
    candidate_ids = {str(rec.get('imdb_id') or rec.get('title', '')).strip().lower()
                     for rec in rec_content if rec.get('imdb_id') or rec.get('title')}
    if not candidate_ids:
        print("No se encontraron candidatos basados en contenido.")
        return []

    # Paso 2: Cargar o calcular las predicciones SVD para el usuario
    cache_dir = ensure_cache_dir()  # Aseguramos que exista el directorio de caché
    cache_file = os.path.join(cache_dir, f"cache_svd_user_{user_id}.pkl")
    if os.path.exists(cache_file):
        rec_svd_df = pd.read_pickle(cache_file)
        print("Predicciones SVD cargadas desde caché.")
    else:
        rec_svd = get_svd_recommendations(user_id, movies_file, ratings_file, top_n=1000)
        rec_svd_df = pd.DataFrame(rec_svd) if isinstance(rec_svd, list) else rec_svd.copy()
        rec_svd_df['title_lower'] = rec_svd_df['title'].str.strip().str.lower()
        rec_svd_df.to_pickle(cache_file)
        print("Predicciones SVD calculadas y guardadas en caché.")

    # Paso 3: Filtrar predicciones SVD que estén en el conjunto de candidatos
    rec_svd_filtered = rec_svd_df[rec_svd_df['title_lower'].isin(candidate_ids)]
    if rec_svd_filtered.empty:
        print("No se encontraron predicciones SVD para los candidatos.")
        return []

    rec_svd_filtered = rec_svd_filtered.sort_values(by='predicted_rating', ascending=False)
    return rec_svd_filtered.head(top_n).to_dict(orient='records')
