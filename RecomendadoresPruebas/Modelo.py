import pickle
import pandas as pd
import numpy as np
import warnings
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm import tqdm
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, KFold
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

#########################################
# Funciones de Preprocesamiento         #
#########################################

def load_preprocessed_data(ratings_filepath: str, movies_filepath: str):
    """ Carga y preprocesa los datos de ratings y películas. """
    ratings_df = pd.read_csv(ratings_filepath, low_memory=False)
    movies_df = pd.read_csv(movies_filepath, low_memory=False)

    # Renombrar 'imdbId' a 'imdb_id' si es necesario
    if 'imdbId' in ratings_df.columns:
        ratings_df.rename(columns={'imdbId': 'imdb_id'}, inplace=True)

    # Asegurar que 'imdb_id' sea numérico en ambos DataFrames
    for df in [movies_df, ratings_df]:
        df['imdb_id'] = pd.to_numeric(df['imdb_id'], errors='coerce')
        df.dropna(subset=['imdb_id'], inplace=True)
        df['imdb_id'] = df['imdb_id'].astype(int)

    return ratings_df, movies_df


#########################################
# Métodos para guardar/cargar modelos   #
#########################################

def save_model(algo, filename="svd_model.pkl"):
    """ Guarda el modelo SVD entrenado en un archivo pickle. """
    with open(filename, "wb") as f:
        pickle.dump(algo, f)
    print(f"✅ Modelo guardado en {filename}")


def load_model(filename="svd_model.pkl"):
    """ Carga un modelo SVD previamente guardado. """
    try:
        with open(filename, "rb") as f:
            algo = pickle.load(f)
        print(f"✅ Modelo cargado desde {filename}")
        return algo
    except FileNotFoundError:
        print("⚠ No se encontró el modelo guardado. Se entrenará uno nuevo...")
        return None


#########################################
# Implementación del Modelo SVD         #
#########################################

def train_svd(data):
    """ Entrena el modelo SVD con hiperparámetros fijos para reducir tiempo de ejecución. """
    algo = SVD(n_factors=100, n_epochs=40, lr_all=0.01, reg_all=0.1)
    trainset = data.build_full_trainset()
    print("⏳ Entrenando modelo SVD...")
    algo.fit(trainset)
    print("✅ Entrenamiento completado.")
    return algo


def evaluate_with_kfold(algo, data, n_splits=5):
    """ Evalúa el modelo SVD con validación cruzada K-Fold. """
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    results = cross_validate(algo, data, measures=['rmse', 'mae'], cv=kf, verbose=True)
    rmse_mean = np.mean(results['test_rmse'])
    mae_mean = np.mean(results['test_mae'])
    print(f"\nPromedio RMSE (K-Fold {n_splits}): {rmse_mean:.4f}")
    print(f"Promedio MAE (K-Fold {n_splits}): {mae_mean:.4f}")
    return results


#########################################
# Generación de Recomendaciones         #
#########################################

def get_svd_recommendations(user_id: int, movies_file: str, ratings_file: str, top_n: int = 10):
    """ Genera recomendaciones para un usuario específico usando SVD. """
    ratings_df, movies_df = load_preprocessed_data(ratings_file, movies_file)

    # Dividir en entrenamiento y prueba
    train_ratings, _ = train_test_split(ratings_df, test_size=0.2, random_state=42)
    data = Dataset.load_from_df(train_ratings[['userId', 'imdb_id', 'rating']], Reader(rating_scale=(0, 5)))

    # Intentar cargar el modelo SVD guardado o entrenarlo si no existe
    algo = load_model()
    if algo is None:
        algo = train_svd(data)
        save_model(algo)

    #evaluate_with_kfold(algo, data)

    # Obtener películas no vistas
    seen_movies = set(train_ratings.loc[train_ratings['userId'] == user_id, 'imdb_id'])
    unseen_movies = [m for m in movies_df['imdb_id'] if m not in seen_movies][:5000]

    print(f"🔄 Generando predicciones para {len(unseen_movies)} películas...")

    predictions = [algo.predict(user_id, mid) for mid in tqdm(unseen_movies)]

    # Crear DataFrame con predicciones
    pred_df = pd.DataFrame([(int(pred.iid), np.clip(pred.est, 0, 5)) for pred in predictions],
                           columns=['imdb_id', 'predicted_rating'])

    movies_df = movies_df.drop_duplicates(subset=['imdb_id'])
    pred_df = pred_df.merge(movies_df[['imdb_id', 'title', 'imdb_votes', 'imdb_rating']], on='imdb_id', how='left')

    pred_df = pred_df.sort_values(by='predicted_rating', ascending=False)

    # Mostrar resultados con nota y número de votos
    print(f"\n🔮 Recomendaciones para el usuario {user_id}:")
    print(pred_df.head(top_n)[['title', 'predicted_rating', 'imdb_rating', 'imdb_votes']].to_string(index=False))

    return pred_df.head(top_n)


#########################################
# Ejecución Principal                   #
#########################################

#if __name__ == "__main__":
    #    ratings_file = "C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/ratings.csv"
    #    movies_file = "C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/final_movies.csv"

    #  user_id = 34  # ID de usuario para probar recomendaciones
    #
    #  print("\n🔍 Cargando datos y generando recomendaciones...")
    #  recommendations = get_svd_recommendations(user_id, movies_file, ratings_file)

# print("\n✅ Proceso finalizado!")
