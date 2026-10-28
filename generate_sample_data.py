"""
generate_sample_data.py
------------------------
Creates a SYNTHETIC movies dataset that mimics the structure of the
commonly-used IMDb-style movie ratings dataset for this task:

    Name, Year, Duration, Genre, Votes, Director, Actor 1, Actor 2, Actor 3, Rating

  - Genre        : one or more genres, comma-separated (e.g. "Action, Drama")
  - Director     : the movie's director
  - Actor 1/2/3  : the three lead actors
  - Votes        : number of user votes the movie received
  - Rating       : the target -- average user/critic rating, 1-10 scale

>>> IMPORTANT FOR YOUR INTERNSHIP SUBMISSION <<<
For your actual CodSoft submission, you can either:
  (a) use this synthetic data as-is (the pipeline works end-to-end with
      it, and the patterns it contains -- some directors/actors/genres
      consistently get higher ratings than others -- are realistic), or
  (b) download a real movie-ratings dataset (search "IMDb Movies India
      dataset" / "IMDb 5000 movie dataset" on Kaggle) and place it at
      data/movies.csv instead, making sure the columns are named the same
      as above (rename columns in the CSV if needed).

Run directly to generate the file:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd
import os

GENRES = [
    "Action", "Drama", "Comedy", "Thriller", "Romance", "Sci-Fi",
    "Horror", "Animation", "Crime", "Adventure", "Fantasy", "Mystery",
    "Biography", "Musical", "Documentary",
]

# Give each genre a slight baseline "quality" effect so there's real signal
# in the data (e.g. Documentary/Biography skew a bit higher, Horror a bit lower)
GENRE_BASELINE = {
    "Action": 0.0, "Drama": 0.4, "Comedy": -0.1, "Thriller": 0.1,
    "Romance": 0.0, "Sci-Fi": 0.1, "Horror": -0.4, "Animation": 0.3,
    "Crime": 0.2, "Adventure": 0.0, "Fantasy": 0.0, "Mystery": 0.1,
    "Biography": 0.5, "Musical": -0.1, "Documentary": 0.6,
}

DIRECTOR_FIRST = ["James", "Maria", "Wei", "Aiko", "Carlos", "Fatima", "Liam",
                   "Sophia", "Raj", "Elena", "Omar", "Grace", "Hiro", "Nadia",
                   "Lucas", "Priya", "Anders", "Chloe", "Diego", "Mei"]
DIRECTOR_LAST = ["Stone", "Reyes", "Chen", "Tanaka", "Silva", "Khan", "Murphy",
                  "Novak", "Patel", "Ivanova", "Haddad", "Walsh", "Sato",
                  "Kovac", "Ramirez", "Sharma", "Berg", "Dupont", "Alvarez", "Lin"]

ACTOR_FIRST = ["Alex", "Jordan", "Sam", "Riley", "Taylor", "Morgan", "Casey",
               "Jamie", "Drew", "Avery", "Quinn", "Reese", "Skyler", "Rowan",
               "Sage", "Dakota", "Phoenix", "River", "Emerson", "Hayden",
               "Blake", "Cameron", "Devon", "Ellis", "Finley"]
ACTOR_LAST = ["Hart", "Bishop", "Vance", "Cross", "Pierce", "Wells", "Hayes",
              "Frost", "Banks", "Lowe", "Marsh", "Reed", "Stark", "Voss",
              "Nash", "Carver", "Doyle", "Knox", "Pike", "Shaw", "Brennan",
              "Fenwick", "Lockhart", "Sutton", "Maddox"]


def _make_name_pool(firsts, lasts, n, rng):
    pool = set()
    while len(pool) < n:
        pool.add(f"{rng.choice(firsts)} {rng.choice(lasts)}")
    return list(pool)


def generate_synthetic_movies(n_movies=1200, random_state=42):
    rng = np.random.RandomState(random_state)

    directors = _make_name_pool(DIRECTOR_FIRST, DIRECTOR_LAST, 60, rng)
    actors = _make_name_pool(ACTOR_FIRST, ACTOR_LAST, 150, rng)

    # Give each director/actor a hidden "skill" level so the model has
    # real signal to learn (just like in real life, some directors/actors
    # are reliably associated with better-rated films)
    director_skill = {d: rng.normal(0, 0.6) for d in directors}
    actor_skill = {a: rng.normal(0, 0.5) for a in actors}

    rows = []
    for i in range(n_movies):
        n_genres = rng.choice([1, 2, 3], p=[0.4, 0.45, 0.15])
        movie_genres = list(rng.choice(GENRES, size=n_genres, replace=False))

        director = rng.choice(directors)
        cast = rng.choice(actors, size=3, replace=False)

        year = int(rng.randint(1980, 2025))
        duration = int(rng.normal(115, 20))
        duration = max(70, min(220, duration))

        # Popularity / votes: log-normal, loosely tied to rating later
        base_votes = rng.lognormal(mean=8.5, sigma=1.6)

        genre_effect = np.mean([GENRE_BASELINE[g] for g in movie_genres])
        director_effect = director_skill[director]
        actor_effect = np.mean([actor_skill[a] for a in cast])
        year_effect = (year - 2000) * 0.002          # tiny upward drift over time
        duration_effect = -abs(duration - 110) * 0.004  # sweet spot around ~110 min

        noise = rng.normal(0, 0.55)

        rating = (
            6.3
            + genre_effect
            + director_effect
            + actor_effect
            + year_effect
            + duration_effect
            + noise
        )
        rating = float(np.clip(rating, 1.0, 9.9))

        votes = int(max(50, base_votes * (1 + 0.15 * (rating - 6.3))))

        rows.append({
            "Name": f"Movie_{i+1:04d}",
            "Year": year,
            "Duration": duration,
            "Genre": ", ".join(movie_genres),
            "Votes": votes,
            "Director": director,
            "Actor 1": cast[0],
            "Actor 2": cast[1],
            "Actor 3": cast[2],
            "Rating": round(rating, 1),
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_movies()
    out_path = os.path.join("data", "movies_sample.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset written to: {out_path}")
    print(f"Shape: {df.shape}")
    print(df.head())
    print("\nRating distribution:")
    print(df["Rating"].describe())
