import pandas as pd
import numpy as np
import warnings
import pickle
import re
import datetime
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from annoy import AnnoyIndex
from scipy.sparse import coo_matrix, csr_matrix

warnings.filterwarnings("ignore")

#########################################
# Funciones de Preprocesamiento         #
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

def load_preprocessed_data(ratings_filepath: str, movies_filepath: str):
    ratings_df = pd.read_csv(ratings_filepath, low_memory=False)
    movies_df = pd.read_csv(movies_filepath, low_memory=False)
    if 'imdbId' in ratings_df.columns:
        ratings_df.rename(columns={'imdbId': 'imdb_id'}, inplace=True)
    for df in [movies_df, ratings_df]:
        df['imdb_id'] = df['imdb_id'].apply(
            lambda x: int(re.sub(r'\D', '', str(x))) if pd.notnull(x) else x)
        df.dropna(subset=['imdb_id'], inplace=True)
        df['imdb_id'] = df['imdb_id'].astype(int)
    # Extraer el año a partir del campo "release_date" (si existe) y crear la columna "release_year"
    if 'release_date' in movies_df.columns:
        movies_df['release_year'] = movies_df['release_date'].apply(lambda d: extract_year(d))
    return ratings_df, movies_df

def filter_ratings(ratings_df: pd.DataFrame, min_user_ratings: int = 20, min_item_ratings: int = 20) -> pd.DataFrame:
    user_counts = ratings_df.groupby('userId').size()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
    item_counts = ratings_df.groupby('imdb_id').size()
    valid_items = item_counts[item_counts >= min_item_ratings].index
    return ratings_df[ratings_df['imdb_id'].isin(valid_items)]

def create_sparse_user_item_matrix(ratings_df: pd.DataFrame):
    # Crear mapeos de usuarios y películas
    unique_users = ratings_df['userId'].unique()
    unique_items = ratings_df['imdb_id'].unique()
    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {i: j for j, i in enumerate(unique_items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {j: i for i, j in item2idx.items()}
    rows = ratings_df['userId'].map(user2idx).values
    cols = ratings_df['imdb_id'].map(item2idx).values
    data = ratings_df['rating'].values
    sparse_matrix = coo_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_items))).tocsr()
    return sparse_matrix, user2idx, item2idx, idx2user, idx2item

def compute_user_stats(sparse_matrix: csr_matrix):
    # Calcular media por usuario
    means = np.array(sparse_matrix.mean(axis=1)).ravel()
    # Calcular la media de los cuadrados
    sq = sparse_matrix.multiply(sparse_matrix)
    mean_sq = np.array(sq.mean(axis=1)).ravel()
    stds = np.sqrt(mean_sq - means ** 2)
    stds[stds == 0] = 1
    return means, stds

def normalize_sparse_matrix(sparse_matrix: csr_matrix, means, stds):
    coo = sparse_matrix.tocoo()
    # Para cada elemento no nulo, normalizar: (valor - media del usuario)/std del usuario
    coo.data = (coo.data - means[coo.row]) / stds[coo.row]
    return coo.tocsr()

def precompute_topk_with_annoy_sparse(normalized_matrix: csr_matrix, k: int, n_components: int = 50, n_trees: int = 10):
    # Aplicar TruncatedSVD directamente sobre la matriz dispersa normalizada
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(normalized_matrix)
    d = n_components
    index = AnnoyIndex(d, 'angular')
    num_users = reduced.shape[0]
    for i in range(num_users):
        index.add_item(i, reduced[i])
    index.build(n_trees)
    topk_sim = {}
    for i in range(num_users):
        neighbors_idx = index.get_nns_by_item(i, k + 1)
        neighbors_idx = [j for j in neighbors_idx if j != i][:k]
        sims = []
        vec = reduced[i]
        for j in neighbors_idx:
            other_vec = reduced[j]
            sim = np.dot(vec, other_vec) / (np.linalg.norm(vec) * np.linalg.norm(other_vec))
            sims.append(sim)
        topk_sim[i] = (neighbors_idx, sims)
    return topk_sim, svd, reduced

#########################################
# Implementación del Modelo PearsonCF   #
# (utilizando matriz dispersa)           #
#########################################
class PearsonCF_sparse:
    def __init__(self, ratings_df: pd.DataFrame, k: int = 30, m_threshold: int = 200,
                 n_components: int = 50, n_trees: int = 10, force_recompute: bool = False):
        self.ratings_df = ratings_df
        self.k = k
        self.m_threshold = m_threshold
        self.n_components = n_components
        self.n_trees = n_trees
        self.sparse_matrix, self.user2idx, self.item2idx, self.idx2user, self.idx2item = self.load_or_create_sparse_matrix(
            force_recompute)
        self.means, self.stds = compute_user_stats(self.sparse_matrix)
        self.normalized_matrix = normalize_sparse_matrix(self.sparse_matrix, self.means, self.stds)
        self.topk_sim, self.svd_model, self.reduced = self.load_or_compute_neighbors(force_recompute)

    def load_or_create_sparse_matrix(self, force_recompute: bool = False):
        if not force_recompute:
            try:
                with open("sparse_matrix.pkl", "rb") as f:
                    sparse_matrix = pickle.load(f)
                with open("user2idx.pkl", "rb") as f:
                    user2idx = pickle.load(f)
                with open("item2idx.pkl", "rb") as f:
                    item2idx = pickle.load(f)
                with open("idx2user.pkl", "rb") as f:
                    idx2user = pickle.load(f)
                with open("idx2item.pkl", "rb") as f:
                    idx2item = pickle.load(f)
                return sparse_matrix, user2idx, item2idx, idx2user, idx2item
            except:
                pass
        sparse_matrix, user2idx, item2idx, idx2user, idx2item = create_sparse_user_item_matrix(self.ratings_df)
        with open("sparse_matrix.pkl", "wb") as f:
            pickle.dump(sparse_matrix, f)
        with open("user2idx.pkl", "wb") as f:
            pickle.dump(user2idx, f)
        with open("item2idx.pkl", "wb") as f:
            pickle.dump(item2idx, f)
        with open("idx2user.pkl", "wb") as f:
            pickle.dump(idx2user, f)
        with open("idx2item.pkl", "wb") as f:
            pickle.dump(idx2item, f)
        return sparse_matrix, user2idx, item2idx, idx2user, idx2item

    def load_or_compute_neighbors(self, force_recompute: bool = False):
        if not force_recompute:
            try:
                with open("topk_annoy_sparse.pkl", "rb") as f:
                    return pickle.load(f)
            except:
                pass
        topk_sim, svd_model, reduced = precompute_topk_with_annoy_sparse(self.normalized_matrix, self.k,
                                                                         self.n_components, self.n_trees)
        with open("topk_annoy_sparse.pkl", "wb") as f:
            pickle.dump((topk_sim, svd_model, reduced), f)
        return topk_sim, svd_model, reduced

    def recommend_movies(self, user: int, movies_df: pd.DataFrame, top_n: int = 10) -> list:
        if user not in self.user2idx:
            print(f"El usuario {user} no se encuentra.")
            return []
        user_idx = self.user2idx[user]
        # Obtener los índices de películas que el usuario ha calificado (filas no nulas)
        row = self.sparse_matrix.getrow(user_idx)
        rated_item_idxs = row.indices
        rated_movie_ids = {self.idx2item[i] for i in rated_item_idxs}
        all_movie_ids = set(movies_df['imdb_id'].unique())
        unseen_movie_ids = list(all_movie_ids - rated_movie_ids)
        predictions = []
        for movie in unseen_movie_ids:
            if movie not in self.item2idx:
                continue
            movie_idx = self.item2idx[movie]
            pred = self.predict_rating(user_idx, movie_idx)
            predictions.append((movie, pred))
        pred_df = pd.DataFrame(predictions, columns=['imdb_id', 'predicted_rating']).dropna()
        pred_df = pred_df.merge(movies_df[['imdb_id', 'title', 'release_year', 'imdb_votes', 'imdb_rating', 'poster_path']],
                                on='imdb_id', how='left')
        pred_df = pred_df[pred_df['imdb_votes'] >= self.m_threshold]
        pred_df = pred_df.sort_values(by='predicted_rating', ascending=False).head(top_n)
        return pred_df.to_dict(orient='records')

    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        if user_idx not in self.topk_sim:
            return np.nan
        neighbors_idxs, sims = self.topk_sim[user_idx]
        if len(neighbors_idxs) == 0:
            return np.nan
        ratings = []
        for n_idx in neighbors_idxs:
            r = self.sparse_matrix[n_idx, movie_idx]
            ratings.append(r)
        ratings = np.array(ratings).ravel()
        sims = np.array(sims)
        valid = ratings > 3
        if valid.sum() == 0:
            return np.nan
        numerator = np.dot(sims[valid], ratings[valid])
        denominator = np.sum(np.abs(sims[valid]))
        return numerator / denominator if denominator != 0 else np.nan

class PearsonCFWrapper:
    def __init__(self, ratings_df: pd.DataFrame, **kwargs):
        self.model = PearsonCF_sparse(ratings_df, **kwargs)

    def recommend_movies(self, user: int, movies_df: pd.DataFrame, top_n: int = 10) -> list:
        return self.model.recommend_movies(user, movies_df, top_n)

def get_user_based_recommendations(user_id: int, movies_file: str, ratings_file: str,
                                   top_n: int = 10, force_recompute: bool = True):
    ratings_df, movies_df = load_preprocessed_data(ratings_file, movies_file)
    ratings_df = filter_ratings(ratings_df, min_user_ratings=20, min_item_ratings=20)
    num_ratings = ratings_df[ratings_df['userId'] == user_id].shape[0]
    print(f"Número de ratings para el usuario {user_id}: {num_ratings}")
    train_ratings, _ = train_test_split(ratings_df, test_size=0.2, random_state=42)
    model = PearsonCFWrapper(train_ratings, force_recompute=force_recompute)
    return model.recommend_movies(user_id, movies_df, top_n)
