import os
import re
import sqlite3

import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_caching import Cache
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from RecomendadoresPruebas.Contenido import get_content_based_recommendations
from RecomendadoresPruebas.Hybrid import get_hybrid_recommendations_cascade
from RecomendadoresPruebas.Items import get_item_based_recommendations
from RecomendadoresPruebas.Usuarios import get_user_based_recommendations

# Configuración inicial de la app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configuración del Login Manager
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Configuración de caché con Redis
app.config['CACHE_TYPE'] = 'redis'
app.config['CACHE_REDIS_URL'] = 'redis://localhost:6379/0'
app.config['CACHE_DEFAULT_TIMEOUT'] = 12000  # 20 minutos
cache = Cache(app)

# Rutas de los archivos de datos
MOVIES_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/final_movies.csv'
RATINGS_FILE = 'C:/Users/tgtob/TFG/ConjutoDatos/DatosFinales/ratings.csv'

# Cargar la clave de la API desde .env
load_dotenv()  # Carga las variables del archivo .env
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
print("TMDB_API_KEY cargada:", TMDB_API_KEY)  # Verifica que se cargue correctamente

def get_tmdb_poster_by_title(title: str) -> str | None:
    """
    Dado un título de película, consulta la API de TMDB para obtener el poster_path
    y retorna la URL completa del póster.
    """
    if not TMDB_API_KEY:
        raise ValueError("La API Key de TMDB no está configurada.")

    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("Error en la consulta a TMDB:", response.status_code)
        return None

    data = response.json()
    if data.get("results"):
        movie = data["results"][0]
        poster_path = movie.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None


def load_preprocessed_data(ratings_filepath: str, movies_filepath: str):
    ratings_df = pd.read_csv(ratings_filepath, low_memory=False)
    movies_df = pd.read_csv(movies_filepath, low_memory=False)
    ratings_df.rename(columns={'imdbId': 'imdb_id'}, inplace=True)
    for df in [movies_df, ratings_df]:
        # Conservamos el imdb_id como string para búsquedas por título
        df['imdb_id'] = df['imdb_id'].astype(str)
        # Si lo deseas, filtra filas que no comiencen con "tt"
        df.drop(df[~df['imdb_id'].str.startswith("tt")].index, inplace=True)
    return ratings_df, movies_df


# Configuración de la base de datos SQLite
DATABASE = './database/users.db'


def init_db():
    if not os.path.exists('./database'):
        os.makedirs('./database')
    conn = sqlite3.connect(DATABASE)
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
    conn.close()


def load_ratings_users():
    ratings_df = pd.read_csv(RATINGS_FILE)
    unique_user_ids = ratings_df['userId'].unique()
    conn = sqlite3.connect(DATABASE)
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
    conn.close()
    print(f"✅ Usuarios insertados desde ratings: {inserted_count}")


@cache.cached(timeout=12000, key_prefix="best_movies_by_genre")
def get_best_movies_by_genre(movies_file: str, genres: list, top_n: int = 30) -> dict:
    df = pd.read_csv(movies_file, low_memory=False)
    df['genres'] = df['genres'].fillna('')
    movies_by_genre = {}
    for genre in genres:
        df_genre = df[df['genres'].str.contains(genre, case=False, na=False)]
        df_genre = df_genre.sort_values(by='imdb_votes', ascending=False).head(top_n)
        movies_by_genre[genre] = df_genre.to_dict('records')
    return movies_by_genre


def get_movie_details(movie_title, movies_file):
    df_movies = pd.read_csv(movies_file, low_memory=False)
    movie_row = df_movies[df_movies['title'].str.lower() == movie_title.lower()]
    if movie_row.empty:
        return None
    movie = movie_row.iloc[0].to_dict()

    # Buscar el póster por título
    poster_url = get_tmdb_poster_by_title(movie.get('title', ''))

    genres_str = movie.get('genres', '')
    genres_list = [g.strip() for g in genres_str.split(',')] if genres_str else []

    movie_details = {
        'title': movie.get('title', 'Título desconocido'),
        'release_year': movie.get('release_year', 'Desconocido'),
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
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
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
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        if input_value.isdigit():
            cursor.execute("SELECT * FROM users WHERE id = ? AND type = 'ratings'", (int(input_value),))
        else:
            cursor.execute("SELECT * FROM users WHERE username = ? AND type = 'manual'", (input_value,))
        user = cursor.fetchone()
        conn.close()
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
        conn = sqlite3.connect(DATABASE)
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
def index():
    # Aquí podrías procesar los datos o simplemente renderizar el template
    return render_template('index.html', username=current_user.username)  # 👈 Ahora enviamos username)


@app.route('/dashboard')
@login_required
def dashboard():
    genres = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
              "Drama", "Family", "Fantasy", "Horror", "Music", "Mystery", "Romance",
              "Science Fiction", "Thriller", "War", "Western"]

    # Se utiliza la función en caché para obtener las mejores películas por género
    movies_by_genre = get_best_movies_by_genre(MOVIES_FILE, genres, top_n=30)

    # Actualizamos cada película para incluir el póster, si no se ha almacenado ya en caché
    for genre, movies in movies_by_genre.items():
        for movie in movies:
            movie['poster_path'] = get_tmdb_poster_by_title(movie.get('title', ''))

    return render_template('dashboard.html', movies_by_genre=movies_by_genre, username=current_user.username)


@app.route('/search')
@login_required
def search():
    # Recoger parámetros de búsqueda y filtros
    query = request.args.get('query', '').strip()
    genre_filter = request.args.get('genre', '').strip()
    year_filter = request.args.get('year', '').strip()
    min_rating_filter = request.args.get('min_rating', '').strip()

    # Si no se recibe ningún parámetro, se devuelve la página sin resultados
    if not query and not genre_filter and not year_filter and not min_rating_filter:
        return render_template('search_results.html', results=[], query=query, username=current_user.username)

    df = pd.read_csv(MOVIES_FILE, low_memory=False)

    # Crear columna 'normalized_title' para facilitar la búsqueda por título
    df['normalized_title'] = df['title'].apply(
        lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x).lower())
    )

    # Filtrar por el término de búsqueda (título)
    norm_query = re.sub(r'[^a-zA-Z0-9]', '', query.lower())
    if norm_query:
        df = df[df['normalized_title'].str.contains(norm_query, na=False)]

    # Aplicar filtro por género
    if genre_filter:
        df = df[df['genres'].str.contains(genre_filter, case=False, na=False)]

    # Aplicar filtro por año
    if year_filter:
        try:
            year_int = int(year_filter)
            df = df[df['release_year'] == year_int]
        except ValueError:
            pass

    # Aplicar filtro por nota mínima
    if min_rating_filter:
        try:
            min_rating_val = float(min_rating_filter)
            df = df[df['imdb_rating'] >= min_rating_val]
        except ValueError:
            pass

    # Seleccionar un subconjunto de resultados para mostrar
    matched = df.head(20)

    # Convertir a diccionario y enriquecer con póster
    results = matched.to_dict('records')
    for item in results:
        item['poster_path'] = get_tmdb_poster_by_title(item.get('title', ''))

    return render_template('search_results.html', results=results, query=query, username=current_user.username)



@app.route('/movie/<movie_title>')
@login_required
def movie_detail(movie_title):
    # Obtén los detalles de la película
    movie_details = get_movie_details(movie_title, MOVIES_FILE)

    # Recomendaciones basadas en contenido (ya existentes)
    recommendations = get_content_based_recommendations(movie_title, MOVIES_FILE, 15)
    for rec in recommendations:
        rec_title = rec.get('title', '')
        rec['poster_path'] = get_tmdb_poster_by_title(rec_title)

    # Se utiliza caché para las recomendaciones híbridas
    # La clave de caché incluirá el user_id y el movie_title (y top_n si es necesario)
    @cache.cached(timeout=12000, key_prefix=lambda: f"hybrid_{current_user.id}_{movie_title}")
    def get_hybrid():
        hybrid_recs = get_hybrid_recommendations_cascade(
            current_user.id,
            movie_title,
            MOVIES_FILE,
            RATINGS_FILE,
            top_n=15,
            candidate_n=10000
        )
        for rec in hybrid_recs:
            rec_title = rec.get('title', '')
            rec['poster_path'] = get_tmdb_poster_by_title(rec_title)
        return hybrid_recs

    hybrid_recommendations = get_hybrid()

    items_recommendations = get_item_based_recommendations(current_user.id, MOVIES_FILE, RATINGS_FILE, top_n=30)
    for rec in items_recommendations:
        rec_title = rec.get('title', '')
        rec['poster_path'] = get_tmdb_poster_by_title(rec_title)

    #users_recommendations = get_user_based_recommendations(current_user.id, MOVIES_FILE, RATINGS_FILE, top_n=30)
    #for rec in users_recommendations:
    #    rec_title = rec.get('title', '')
    #    rec['poster_path'] = get_tmdb_poster_by_title(rec_title)

    return render_template(
        'movie_detail.html',
        movie=movie_details,
        recommendations=recommendations,
        hybrid_recommendations=hybrid_recommendations,
        items_recommendations=items_recommendations,
        #users_recommendations=users_recommendations,
        username=current_user.username
    )


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        init_db()
        load_ratings_users()
    else:
        print("✅ La base de datos ya existe. No se insertarán usuarios duplicados.")

    app.run(debug=True)
