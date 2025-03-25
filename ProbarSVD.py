import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, KFold
from sklearn.model_selection import train_test_split
from RecomendadoresPruebas.Modelo import load_preprocessed_data, train_svd, evaluate_with_kfold
import warnings

warnings.filterwarnings("ignore")

# Rutas a los archivos de datos
MOVIES_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_movies.csv'
RATINGS_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_ratings.csv'

# Cargar los datos de ratings y películas
ratings_df, movies_df = load_preprocessed_data(RATINGS_FILE, MOVIES_FILE)

# Para acelerar la evaluación, se toma una muestra aleatoria del 1% del dataset de ratings
sample_fraction = 0.5
ratings_df_sample = ratings_df.sample(frac=sample_fraction, random_state=42)

# Configurar el objeto Reader (asumiendo ratings en escala de 0 a 5)
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings_df_sample[['userId', 'imdb_id', 'rating']], reader)

# Entrenar el modelo SVD utilizando la función train_svd
algo = train_svd(data)

# Evaluar el modelo SVD con validación cruzada utilizando KFold (se reduce a 3 particiones para mayor velocidad)
results = evaluate_with_kfold(algo, data, n_splits=3)

# Calcular la media de RMSE y MAE obtenidos en la validación cruzada
mean_rmse = np.mean(results['test_rmse'])
mean_mae = np.mean(results['test_mae'])

print("Evaluación del Modelo SVD (con muestra del 1% y 3 folds):")
print("Mean RMSE: {:.4f}".format(mean_rmse))
print("Mean MAE: {:.4f}".format(mean_mae))
