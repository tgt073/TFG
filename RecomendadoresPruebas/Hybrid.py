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
		# Evitar división por cero si todas las puntuaciones son iguales
		rec[key + '_norm'] = (rec.get(key, 0) - min_val) / (max_val - min_val) if max_val != min_val else 0.5
	return recs


def get_hybrid_recommendations_cascade(user_id, movie_title, movies_file, ratings_file, top_n=20, candidate_n=50):
	"""
	Enfoque en cascada:
	  1. Se generan candidatos mediante el recomendador basado en contenido.
	  2. Se obtienen las predicciones SVD (para un amplio conjunto) y se filtran
		 aquellas cuyo título (normalizado a minúsculas) esté en el conjunto de candidatos.
	  3. Se ordenan por 'predicted_rating' y se devuelven las top_n recomendaciones.

	Devuelve una lista de diccionarios, similar a la función get_content_based_recommendations.
	"""
	# Paso 1: Generar candidatos con el méthodo basado en contenido
	rec_content = get_content_based_recommendations(movie_title, movies_file, num_recommendations=candidate_n)
	candidate_ids = set()
	for rec in rec_content:
		# Se usa 'imdb_id' si está disponible; si no, se utiliza el título
		identifier = rec.get('imdb_id') or rec.get('title')
		if identifier:
			candidate_ids.add(str(identifier).strip().lower())
	if not candidate_ids:
		print("No se encontraron candidatos basados en contenido.")
		return []

	# Paso 2: Obtener predicciones SVD para un conjunto amplio de películas
	rec_svd = get_svd_recommendations(user_id, movies_file, ratings_file, top_n=5000)
	if isinstance(rec_svd, pd.DataFrame):
		rec_svd_df = rec_svd.copy()
	else:
		rec_svd_df = pd.DataFrame(rec_svd)

	# Normalizar los títulos para la comparación
	rec_svd_df['title_lower'] = rec_svd_df['title'].str.strip().str.lower()
	# Filtrar las predicciones SVD que estén en el conjunto de candidatos (según el título)
	rec_svd_filtered = rec_svd_df[rec_svd_df['title_lower'].isin(candidate_ids)]
	if rec_svd_filtered.empty:
		print("No se encontraron predicciones SVD para los candidatos.")
		return []

	# Paso 3: Ordenar las predicciones filtradas por 'predicted_rating'
	rec_svd_filtered = rec_svd_filtered.sort_values(by='predicted_rating', ascending=False)

	# Devolver las top_n recomendaciones como lista de diccionarios
	return rec_svd_filtered.head(top_n).to_dict(orient='records')


