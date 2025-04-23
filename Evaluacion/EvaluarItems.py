import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from RecomendadoresPruebas.Items import load_preprocessed_data, filter_ratings, ItemBasedCF

warnings.filterwarnings("ignore")

# Rutas a los archivos
MOVIES_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_movies.csv'
RATINGS_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_ratings.csv'

# Cargar y filtrar datos
ratings_df, movies_df = load_preprocessed_data(RATINGS_FILE, MOVIES_FILE)
ratings_df = filter_ratings(ratings_df, min_user_ratings=20, min_item_ratings=20)

# Tomar una muestra para acelerar la evaluación
sample_fraction = 0.1
ratings_sample = ratings_df.sample(frac=sample_fraction, random_state=42)

# Configurar KFold para validación cruzada
kf = KFold(n_splits=3, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []

fold = 1
for train_index, test_index in kf.split(ratings_sample):
    train_data = ratings_sample.iloc[train_index]
    test_data = ratings_sample.iloc[test_index]

    # Entrenar modelo basado en ítems
    item_model = ItemBasedCF(train_data, k=20, m_threshold=150, lambda_shrink=10)

    predictions = []
    actuals = []

    for _, row in test_data.iterrows():
        user = row['userId']
        movie = row['imdb_id']
        # Solo se evalúan casos donde el usuario e ítem están en el set de entrenamiento
        if user in train_data['userId'].values and movie in train_data['imdb_id'].values:
            pred = item_model.predict_rating(user, movie)
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

# Mostrar promedios
if rmse_scores:
    print("\nEvaluación del Modelo basado en Ítems (ItemBasedCF) con validación cruzada:")
    print("Mean RMSE: {:.4f}".format(np.mean(rmse_scores)))
    print("Mean MAE: {:.4f}".format(np.mean(mae_scores)))
else:
    print("No se obtuvieron suficientes predicciones válidas para calcular métricas.")
