import os
import re
import sqlite3
import datetime

import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from RecomendadoresPruebas.Contenido import get_content_based_recommendations
from RecomendadoresPruebas.Hybrid import get_hybrid_recommendations_cascade
from RecomendadoresPruebas.Items import get_item_based_recommendations
from RecomendadoresPruebas.Usuarios import get_user_based_recommendations
from RecomendadoresPruebas.Modelo import get_svd_recommendations

# Configuración inicial de la app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configuración del Login Manager
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Rutas de los archivos de datos
MOVIES_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_movies.csv'
RATINGS_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/Final_ratings.csv'

# Cargar la clave de la API desde .env (se usa solo como fallback si no se dispone del poster en el dataset)
load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
print("TMDB_API_KEY cargada:", TMDB_API_KEY)

def extract_year(date_str: str) -> int | None:
    """
    Extrae y retorna el año (entero) a partir de una fecha en formato MM/DD/YYYY.
    Si la conversión falla, retorna None.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.datetime.strptime(date_str, '%m/%d/%Y').year
    except Exception:
        return None

def get_movie_year(movie: dict) -> int | None:
    """
    Retorna el año de la película buscando en 'release_year' (o, si no existe, en 'release_date').
    Si 'release_year' ya es un entero, lo devuelve directamente.
    """
    if 'release_year' in movie:
        # Si es un entero, lo usamos directamente
        if isinstance(movie['release_year'], int):
            return movie['release_year']
        # Si es una cadena, intentamos extraer el año
        elif isinstance(movie['release_year'], str):
            return extract_year(movie['release_year'])
    # Si no hay 'release_year', intentamos con 'release_date'
    if 'release_date' in movie and isinstance(movie['release_date'], str):
        return extract_year(movie['release_date'])
    return None


def get_tmdb_poster_by_title(title: str, year: int | None = None) -> str | None:
    """
    Consulta la API de TMDB para obtener el poster_path usando título y opcionalmente año,
    y retorna la URL completa del póster. Se usa como fallback si no se dispone del poster_path.
    """
    if not TMDB_API_KEY:
        raise ValueError("La API Key de TMDB no está configurada.")
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    if year:
        params["year"] = year
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("Error en la consulta a TMDB:", response.status_code)
        return None
    data = response.json()
    results = data.get("results", [])
    if results:
        poster_path = results[0].get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

def get_full_poster_url(poster_path: str) -> str:
    """Construye la URL completa del póster a partir del poster_path almacenado en el dataset."""
    return f"https://image.tmdb.org/t/p/w500{poster_path}"

def obtener_poster(movie: dict) -> str:
    """
    Retorna la URL completa del póster usando primero el poster_path del dataset.
    Si no existe o es inválido, se usa la API de TMDB como fallback, usando el título y año.
    """
    poster_field = movie.get('poster_path')
    if isinstance(poster_field, str) and poster_field.strip():
        return get_full_poster_url(poster_field)
    else:
        title = movie.get('title', '')
        year = get_movie_year(movie)
        return get_tmdb_poster_by_title(title, year) or ""

def load_preprocessed_data(ratings_filepath: str, movies_filepath: str):
    ratings_df = pd.read_csv(ratings_filepath, low_memory=False)
    movies_df = pd.read_csv(movies_filepath, low_memory=False)
    ratings_df.rename(columns={'imdbId': 'imdb_id'}, inplace=True)
    for df in [movies_df, ratings_df]:
        df['imdb_id'] = df['imdb_id'].astype(str)
        df.drop(df[~df['imdb_id'].str.startswith("tt")].index, inplace=True)
    return ratings_df, movies_df

# Bases de datos
USER_DATABASE = './database/users.db'
MOVIES_DATABASE = './database/movies_genre.db'

def init_db():
    """Crea la base de datos y las tablas si no existen."""
    if not os.path.exists('./database'):
        os.makedirs('./database')
    with sqlite3.connect(USER_DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT,
                type TEXT NOT NULL CHECK (type IN ('ratings', 'manual'))
            )
        ''')
        conn.commit()
    with sqlite3.connect(MOVIES_DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_recommendations (
                genre TEXT,
                movie_title TEXT,
                imdb_id TEXT,
                poster_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (genre, imdb_id)
            )
        ''')
        conn.commit()
    print("✅ Base de datos `movies_genre.db` creada correctamente.")

def save_global_recommendations(recommendations):
    """Guarda en la base de datos las recomendaciones globales si aún no existen."""
    with sqlite3.connect(MOVIES_DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM global_recommendations")
        if cursor.fetchone()[0] > 0:
            print("⚠ Las recomendaciones ya existen en la base de datos. No se insertarán duplicados.")
            return
        for genre, movies in recommendations.items():
            if not genre.strip():
                print("⚠ Se detectó un género vacío, no se guardará en la base de datos.")
                continue
            if not movies:
                print(f"⚠ No hay películas para el género: {genre}, no se guardará en la base de datos.")
                continue
            print(f"✅ Guardando películas para el género: {genre}")
            for movie in movies:
                cursor.execute("""
                    INSERT OR IGNORE INTO global_recommendations (genre, movie_title, imdb_id, poster_path)
                    VALUES (?, ?, ?, ?)
                """, (genre, movie['title'], movie.get('imdb_id', ''), movie.get('poster_path', '')))
        conn.commit()
    print("✅ Recomendaciones guardadas en `movies_genre.db`.")

def get_global_recommendations():
    """Recupera las recomendaciones globales almacenadas en la base de datos."""
    with sqlite3.connect(MOVIES_DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT genre, movie_title, imdb_id, poster_path FROM global_recommendations ORDER BY genre ASC")
        rows = cursor.fetchall()
    movies_by_genre = {}
    for genre, title, imdb_id, poster_path in rows:
        if genre not in movies_by_genre:
            movies_by_genre[genre] = []
        movies_by_genre[genre].append({'title': title, 'imdb_id': imdb_id, 'poster_path': poster_path})
    return dict(sorted(movies_by_genre.items()))

def load_ratings_users():
    ratings_df = pd.read_csv(RATINGS_FILE)
    unique_user_ids = ratings_df['userId'].unique()
    with sqlite3.connect(USER_DATABASE) as conn:
        cursor = conn.cursor()
        inserted_count = 0
        for user_id in unique_user_ids:
            cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO users (id, username, password, type) VALUES (?, ?, ?, ?)",
                    (int(user_id), f"user_{int(user_id)}", None, 'ratings')
                )
                inserted_count += 1
        conn.commit()
    print(f"✅ Usuarios insertados desde ratings: {inserted_count}")

def get_best_movies_by_genre(movies_file: str, genres: list, top_n: int = 30) -> dict:
    df = pd.read_csv(movies_file, low_memory=False)
    df['genres'] = df['genres'].fillna('')
    # Umbral mínimo de votos: percentil 90
    m = df['imdb_votes'].quantile(0.10)
    movies_by_genre = {}
    for genre in genres:
        df_genre = df[df['genres'].str.contains(genre, case=False, na=False)]
        df_genre = df_genre[df_genre['imdb_votes'] >= m]
        df_genre = df_genre.sort_values(by='imdb_votes', ascending=False).head(top_n)
        movies_by_genre[genre] = df_genre.to_dict('records')
    return movies_by_genre

def get_movie_details(movie_title, movies_file):
    df_movies = pd.read_csv(movies_file, low_memory=False)
    movie_row = df_movies[df_movies['title'].str.lower() == movie_title.lower()]
    if movie_row.empty:
        return None
    movie = movie_row.iloc[0].to_dict()
    year_int = extract_year(movie.get('release_date'))
    # Usar el poster_path del dataset si existe; de lo contrario, fallback a la API
    poster_url = obtener_poster(movie)
    genres_str = movie.get('genres', '')
    genres_list = [g.strip() for g in genres_str.split(',')] if genres_str else []
    movie_details = {
        'title': movie.get('title', 'Título desconocido'),
        'release_year': year_int if year_int is not None else 'Desconocido',
        'overview': movie.get('overview', 'Sin descripción disponible.'),
        'genres': genres_list,
        'director': movie.get('director', 'Desconocido'),
        'main_actors': movie.get('main_actors', '').split(', '),
        'imdb_rating': movie.get('imdb_rating', 'N/A'),
        'imdb_votes': movie.get('imdb_votes', 0),
        'poster_path': poster_url
    }
    return movie_details

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    with sqlite3.connect(USER_DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
    if user:
        return User(user[0], user[1], user[2])
    return None

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        input_value = request.form['username']
        password = request.form['password']
        with sqlite3.connect(USER_DATABASE) as conn:
            cursor = conn.cursor()
            if input_value.isdigit():
                cursor.execute("SELECT * FROM users WHERE id = ? AND type = 'ratings'", (int(input_value),))
            else:
                cursor.execute("SELECT * FROM users WHERE username = ? AND type = 'manual'", (input_value,))
            user = cursor.fetchone()
        if user:
            if user[3] == 'ratings' or (user[3] == 'manual' and check_password_hash(user[2], password)):
                user_obj = User(user[0], user[1], user[2])
                login_user(user_obj)
                return redirect(url_for('dashboard'))
            else:
                flash('Contraseña incorrecta.')
        else:
            flash('Usuario no encontrado.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'], method='sha256')
        with sqlite3.connect(USER_DATABASE) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                flash('Registro exitoso. Inicia sesión.')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('El usuario ya existe.')
        conn.close()
    return render_template('register.html')

@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    recommendations = []
    error_message = None

    if request.method == 'POST':
        method = request.form.get('method')
        user_id = request.form.get('user_id', type=int, default=current_user.id)
        movie_title = request.form.get('movie_title', '').strip()
        num_recommendations = request.form.get('num_recommendations', type=int, default=10)
        try:
            if method == 'colaborativo_usuarios':
                recommendations = get_user_based_recommendations(user_id, MOVIES_FILE, RATINGS_FILE,
                                                                 top_n=num_recommendations)
            elif method == 'colaborativo_items':
                recommendations = get_item_based_recommendations(user_id, MOVIES_FILE, RATINGS_FILE,
                                                                 top_n=num_recommendations)
            elif method == 'svd':
                recommendations = get_svd_recommendations(user_id, MOVIES_FILE, RATINGS_FILE, top_n=num_recommendations)
            elif method == 'contenido':
                if not movie_title:
                    error_message = "Debe proporcionar un título de película para la recomendación basada en contenido."
                else:
                    recommendations = get_content_based_recommendations(movie_title, MOVIES_FILE,
                                                                        num_recommendations=num_recommendations)
            elif method == 'hibrido':
                if not movie_title:
                    error_message = "Debe proporcionar un título de película para la recomendación híbrida."
                else:
                    recommendations = get_hybrid_recommendations_cascade(user_id, movie_title, MOVIES_FILE,
                                                                         RATINGS_FILE, top_n=num_recommendations)
        except Exception as e:
            error_message = f"Ocurrió un error: {str(e)}"

    if recommendations:
        for rec in recommendations:
            rec['poster_path'] = obtener_poster(rec)

    return render_template('index.html', username=current_user.username, recommendations=recommendations,
                           error_message=error_message)

def get_all_genres():
    """Extrae todos los géneros únicos desde el dataset y los ordena alfabéticamente."""
    df = pd.read_csv(MOVIES_FILE, low_memory=False)
    df['genres'] = df['genres'].fillna('')
    all_genres = set()
    for genre_list in df['genres']:
        genres = [g.strip() for g in genre_list.split(',') if g.strip()]
        all_genres.update(genres)
    if "" in all_genres:
        all_genres.remove("")
    return sorted(list(all_genres))

@app.route('/dashboard')
@login_required
def dashboard():
    """Renderiza el dashboard con las recomendaciones globales."""
    recommendations = get_global_recommendations()
    if not recommendations:
        genres = get_all_genres()
        # Aquí llamamos directamente a get_best_movies_by_genre para obtener las más populares
        movies_by_genre = get_best_movies_by_genre(MOVIES_FILE, genres, top_n=30)

        # Para cada película, asignamos su póster (dataset o fallback)
        for genre, movies in movies_by_genre.items():
            for movie in movies:
                movie['poster_path'] = obtener_poster(movie)

        # Guardamos en la base de datos para no recalcular siempre
        save_global_recommendations(movies_by_genre)
        recommendations = movies_by_genre

    return render_template('dashboard.html', movies_by_genre=recommendations, username=current_user.username)

def filter_movies(df: pd.DataFrame,
                  year_min: int,
                  year_max: int,
                  rating_min: float,
                  rating_max: float,
                  genre_filter: str = "") -> pd.DataFrame:
    """
    Aplica los filtros de año (year_min, year_max), rating (rating_min, rating_max)
    y género (genre_filter) sobre el DataFrame df y retorna el subset filtrado.
    Se asume que df tiene columnas 'release_date' (o 'release_year'), 'imdb_rating', 'genres'.
    """
    # 1) Asegúrate de tener la columna 'year' para filtrar por año
    if 'year' not in df.columns:
        # Por ejemplo, extraemos de 'release_date' el año
        df['year'] = df['release_date'].apply(lambda d: extract_year(d))

    # 2) Convertir 'imdb_rating' a numérico
    df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce').fillna(0)

    # 3) Filtro de rango de años
    df = df[(df['year'] >= year_min) & (df['year'] <= year_max)]

    # 4) Filtro de rango de rating
    df = df[(df['imdb_rating'] >= rating_min) & (df['imdb_rating'] <= rating_max)]

    # 5) Filtro de género
    if genre_filter:
        df = df[df['genres'].str.contains(genre_filter, case=False, na=False)]

    return df


def filter_movies(df: pd.DataFrame,
                  year_min: int,
                  year_max: int,
                  rating_min: float,
                  rating_max: float,
                  genre_filter: str = "") -> pd.DataFrame:
    """
    Filtra el DataFrame df según el rango de años, nota y género.
    Se asume que df tiene la columna 'release_date' para extraer el año.
    """
    if 'year' not in df.columns:
        df['year'] = df['release_date'].apply(lambda d: extract_year(d))
    df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce').fillna(0)

    df = df[(df['year'] >= year_min) & (df['year'] <= year_max)]
    df = df[(df['imdb_rating'] >= rating_min) & (df['imdb_rating'] <= rating_max)]
    if genre_filter:
        df = df[df['genres'].str.contains(genre_filter, case=False, na=False)]
    return df


@app.route('/filter')
@login_required
def filter_view():
    """
    Aplica los filtros de año, nota y género. Si no se aplican filtros (se usan los valores por defecto),
    muestra las películas más populares (por ejemplo, las que tienen más votos).
    Además, pasa a la plantilla un diccionario con los filtros aplicados.
    """
    try:
        year_min = int(request.args.get('year_min', 1900))
        year_max = int(request.args.get('year_max', 2024))
    except ValueError:
        year_min, year_max = 1900, 2024

    try:
        rating_min = float(request.args.get('rating_min', 0.0))
        rating_max = float(request.args.get('rating_max', 10.0))
    except ValueError:
        rating_min, rating_max = 0.0, 10.0

    genre_filter = request.args.get('genre', '').strip()

    # Valores por defecto
    default_year_min, default_year_max = 1900, 2024
    default_rating_min, default_rating_max = 0.0, 10.0

    # Armar un diccionario con los filtros aplicados
    applied_filters = {}
    if year_min != default_year_min or year_max != default_year_max:
        applied_filters['Año'] = f"{year_min} - {year_max}"
    if rating_min != default_rating_min or rating_max != default_rating_max:
        applied_filters['Nota'] = f"{rating_min} - {rating_max}"
    if genre_filter:
        applied_filters['Género'] = genre_filter

    df = pd.read_csv(MOVIES_FILE, low_memory=False)
    if applied_filters:
        # Se aplican los filtros
        filtered_df = filter_movies(df, year_min, year_max, rating_min, rating_max, genre_filter)
        filtered_df = filtered_df.sort_values(by='imdb_votes', ascending=False)
    else:
        # Si no se aplicaron filtros, mostramos las películas más populares
        filtered_df = df.copy()
        filtered_df['imdb_votes'] = pd.to_numeric(filtered_df['imdb_votes'], errors='coerce').fillna(0)
        filtered_df = filtered_df.sort_values(by='imdb_votes', ascending=False)

    # Seleccionamos un subconjunto (por ejemplo, 50)
    matched = filtered_df.drop_duplicates(subset=['title']).head(50)
    results = matched.to_dict('records')
    for item in results:
        item['poster_path'] = obtener_poster(item)

    # Si no se aplicaron filtros, se puede definir query="Populares" o dejarlo vacío.
    query = "Filtros aplicados" if applied_filters else "Populares"

    return render_template('search_results.html',
                           results=results,
                           query=query,
                           applied_filters=applied_filters,
                           username=current_user.username)


@app.route('/search')
@login_required
def search():
    query = request.args.get('query', '').strip()
    genre_filter = request.args.get('genre', '').strip()
    year_filter = request.args.get('year', '').strip()
    min_rating_filter = request.args.get('min_rating', '').strip()
    if not query and not genre_filter and not year_filter and not min_rating_filter:
        return render_template('search_results.html', results=[], query=query, username=current_user.username)
    df = pd.read_csv(MOVIES_FILE, low_memory=False)
    df['normalized_title'] = df['title'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x).lower()))
    norm_query = re.sub(r'[^a-zA-Z0-9]', '', query.lower())
    if norm_query:
        df = df[df['normalized_title'].str.contains(norm_query, na=False)]
    if genre_filter:
        df = df[df['genres'].str.contains(genre_filter, case=False, na=False)]
    if year_filter:
        try:
            year_int = int(year_filter)
            # Creamos una columna con el año extraído de 'release_date'
            df['year'] = df['release_date'].apply(lambda d: extract_year(d))
            df = df[df['year'] == year_int]
        except ValueError:
            pass
    if min_rating_filter:
        try:
            min_rating_val = float(min_rating_filter)
            df = df[df['imdb_rating'] >= min_rating_val]
        except ValueError:
            pass
    df = df.drop_duplicates(subset=['title'])
    matched = df.head(20)
    results = matched.to_dict('records')
    for item in results:
        year_int = get_movie_year(item)
        item['poster_path'] = obtener_poster(item)
    return render_template('search_results.html', results=results, query=query, username=current_user.username)

@app.route('/movie/<movie_title>')
@login_required
def movie_detail(movie_title):
    movie_details = get_movie_details(movie_title, MOVIES_FILE)
    recommendations = get_content_based_recommendations(movie_title, MOVIES_FILE, 16)
    for rec in recommendations:
        rec['poster_path'] = obtener_poster(rec)
    def get_hybrid():
        hybrid_recs = get_hybrid_recommendations_cascade(
            current_user.id,
            movie_title,
            MOVIES_FILE,
            RATINGS_FILE,
            top_n=15,
            candidate_n=1000
        )
        for rec in hybrid_recs:
            rec['poster_path'] = obtener_poster(rec)
        return hybrid_recs
    hybrid_recommendations = get_hybrid()
    items_recommendations = get_item_based_recommendations(current_user.id, MOVIES_FILE, RATINGS_FILE, top_n=15)
    for rec in items_recommendations:
        rec['poster_path'] = obtener_poster(rec)
    users_recommendations = get_user_based_recommendations(current_user.id, MOVIES_FILE, RATINGS_FILE, top_n=15)
    for rec in users_recommendations:
        rec['poster_path'] = obtener_poster(rec)
    svd_recommendations = get_svd_recommendations(current_user.id, MOVIES_FILE, RATINGS_FILE, top_n=15)
    for rec in svd_recommendations:
        rec['poster_path'] = obtener_poster(rec)
    return render_template(
        "movie_detail.html",
        movie=movie_details,
        username=current_user.username,
        recommendations=recommendations,
        hybrid_recommendations=hybrid_recommendations,
        items_recommendations=items_recommendations,
        users_recommendations=users_recommendations,
        svd_recommendations=svd_recommendations
    )

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    if not os.path.exists(USER_DATABASE) or not os.path.exists(MOVIES_DATABASE):
        init_db()
    app.run(debug=True)
