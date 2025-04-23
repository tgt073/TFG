import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from RecomendadoresPruebas.Usuarios import load_preprocessed_data, filter_ratings, PearsonCFWrapper

warnings.filterwarnings("ignore")

# Rutas a los archivos
MOVIES_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_movies.csv'
RATINGS_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_ratings.csv'

# Cargar y filtrar datos
ratings_df, movies_df = load_preprocessed_data(RATINGS_FILE, MOVIES_FILE)
ratings_df = filter_ratings(ratings_df, min_user_ratings=20, min_item_ratings=20)

# Tomar una muestra para acelerar evaluación
sample_fraction = 0.1
ratings_sample = ratings_df.sample(frac=sample_fraction, random_state=42)

# Validación cruzada con 3 folds
kf = KFold(n_splits=3, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []

fold = 1
for train_index, test_index in kf.split(ratings_sample):
	train_data = ratings_sample.iloc[train_index]
	test_data = ratings_sample.iloc[test_index]

	# Entrenar modelo Pearson en el fold actual
	user_model = PearsonCFWrapper(train_data, force_recompute=True)

	predictions = []
	actuals = []

	for _, row in test_data.iterrows():
		user = row['userId']
		movie = row['imdb_id']
		if user in user_model.model.user2idx and movie in user_model.model.item2idx:
			user_idx = user_model.model.user2idx[user]
			movie_idx = user_model.model.item2idx[movie]
			pred = user_model.model.predict_rating(user_idx, movie_idx)
			if not np.isnan(pred):
				predictions.append(pred)
				actuals.append(row['rating'])

	if predictions:
		mse = mean_squared_error(actuals, predictions)
		rmse = np.sqrt(mse)
		mae = mean_absolute_error(actuals, predictions)
		rmse_scores.append(rmse)
		mae_scores.append(mae)
		print(f"Fold {fold}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

	else:
		print(f"Fold {fold}: No se obtuvieron predicciones válidas.")
	fold += 1

# Mostrar resultados promedio
if rmse_scores:
	print("\nEvaluación del Modelo basado en Usuarios (PearsonCF) con validación cruzada:")
	print("Mean RMSE: {:.4f}".format(np.mean(rmse_scores)))
	print("Mean MAE: {:.4f}".format(np.mean(mae_scores)))
else:
	print("No se obtuvieron suficientes predicciones válidas para calcular métricas.")
