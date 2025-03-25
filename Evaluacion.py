import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold as SklearnKFold

# Importar tus implementaciones propias
from RecomendadoresPruebas.Modelo import load_preprocessed_data
from RecomendadoresPruebas.Usuarios import PearsonCF_sparse
from RecomendadoresPruebas.Items import ItemBasedCF

# Rutas a los archivos
MOVIES_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_movies.csv'
RATINGS_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_ratings.csv'

# Cargar datos preprocesados
ratings_df, movies_df = load_preprocessed_data(RATINGS_FILE, MOVIES_FILE)
ratings_df_filtered = ratings_df.groupby('userId').filter(lambda x: len(x) >= 20)

# Dataset para Surprise (SVD)
from surprise import Reader, Dataset, SVD
reader = Reader(rating_scale=(0, 5))
data_surprise = Dataset.load_from_df(ratings_df_filtered[['userId', 'imdb_id', 'rating']], reader)

# Evaluación SVD (usando librería Surprise)
print("\n🔹 Evaluando modelo SVD (Surprise):")
algo_svd = SVD(n_factors=100, n_epochs=40, lr_all=0.01, reg_all=0.1, random_state=42)

results_svd = cross_validate(algo_svd, data_surprise, measures=['rmse', 'mae'], cv=5, verbose=True)
print(f"\n📊 Promedio RMSE (SVD): {np.mean(results_svd['test_rmse']):.4f}")
print(f"📊 Promedio MAE SVD: {np.mean(results_svd['test_mae']):.4f}")

# Función genérica para evaluar Usuarios e Ítems con KFold
def evaluate_cf_kfold(model_class, ratings_df, n_splits=5):
    kf = SklearnKFold(n_splits=n_splits, random_state=42, shuffle=True)
    actual, predicted = [], []

    for fold, (train_index, test_index) in enumerate(kf.split(ratings_df)):
        print(f"\n🔍 Fold {fold + 1}/{n_splits}")
        train_ratings, test_ratings = ratings_df.iloc[train_index], ratings_df.iloc[test_index]
        model = model_class(train_ratings)

        for _, row in test_ratings.iterrows():
            pred_rating = model.predict_rating(row['userId'], row['imdb_id'])
            if not np.isnan(pred_rating):
                actual.append(row['rating'])
                predicted.append(pred_rating)

    if len(actual) == 0:
        print("❌ No se obtuvieron predicciones válidas.")
        return None, None

    rmse = mean_squared_error(actual, predicted, squared=False)
    mae = mean_absolute_error(actual, predicted)

    print(f"\n📊 Promedios tras {n_splits}-Fold:")
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ MAE:  {mae:.4f}")

    return rmse, mae

print("\n🔹 Evaluando Filtrado Colaborativo basado en Usuarios (PearsonCF)")
#evaluate_cf_kfold(PearsonCF, ratings_df_filtered, n_splits=3)
print("\n🔹 Evaluando Filtrado Colaborativo basado en Ítems (ItemBasedCF)")
evaluate_cf_kfold(ItemBasedCF, ratings_df_filtered, n_splits=5)
