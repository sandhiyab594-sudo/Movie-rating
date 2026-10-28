# Movie Rating Prediction with Python — CodSoft Data Science Internship

A complete, runnable machine learning project that predicts a movie's
rating from its **genre, director, and actors** (plus year, duration, and
vote count). It covers everything in the task brief:

- [x] Analyze historical movie data (EDA: rating distribution, average
      rating by genre, correlations)
- [x] Preprocessing & **feature engineering** for categorical data
      (genre / director / actors) — the heart of this task
- [x] Train/test split (done carefully, *before* feature engineering, to
      avoid data leakage — see section 5)
- [x] Train regression models: **Linear Regression** and **Random Forest
      Regressor**
- [x] Evaluate with **MAE, MSE, RMSE, R² score**
- [x] Identify which factors influence ratings the most

---

## 1. Project structure

```
movie_rating_prediction_project/
├── data/
│   └── movies_sample.csv          ← synthetic demo data (auto-generated)
├── outputs/                       ← created automatically when you run the pipeline
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_engineer.pkl
│   ├── rating_distribution.png
│   ├── avg_rating_by_genre.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted_*.png
│   ├── residuals_*.png
│   ├── feature_importance.png
│   └── model_comparison.csv
├── generate_sample_data.py        ← makes a synthetic dataset (for testing)
├── feature_engineering.py         ← target-encoding logic (the trickiest part)
├── movie_rating_prediction.py     ← MAIN script: run this
├── requirements.txt
└── README.md
```

---

## 2. About the dataset

This project expects a CSV with these columns (this matches the popular
IMDb-style movie datasets used for this exact CodSoft task — search "IMDb
Movies India dataset" on Kaggle if you want the real one):

| Column      | Meaning                                              |
|-------------|--------------------------------------------------------|
| `Name`      | Movie title (not used as a feature)                    |
| `Year`      | Release year                                            |
| `Duration`  | Runtime in minutes                                       |
| `Genre`     | One or more genres, comma-separated (e.g. "Action, Drama") |
| `Votes`     | Number of user votes                                     |
| `Director`  | Director's name                                          |
| `Actor 1`   | Lead actor                                                |
| `Actor 2`   | Second lead actor                                         |
| `Actor 3`   | Third lead actor                                          |
| `Rating`    | Target — average user/critic rating (1-10 scale)          |

A synthetic dataset is generated automatically so you can test the whole
pipeline immediately. For your real submission, either use the synthetic
data as-is, or download a real IMDb-style dataset and place it at
`data/movies.csv` with the same column names (rename columns if needed).

---

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. How to run

### Option A — Quick test with synthetic data

```bash
python generate_sample_data.py        # creates data/movies_sample.csv
python movie_rating_prediction.py     # runs the full pipeline on it
```

### Option B — Using your own / a real dataset

```bash
python movie_rating_prediction.py --data data/movies.csv
```

### Optional flags

```bash
python movie_rating_prediction.py --data data/movies.csv --test_size 0.25
```

Check the `outputs/` folder afterwards for all the plots, the model
comparison table, and the saved model files.

---

## 5. The interesting part: encoding Director / Actor / Genre

This is what makes movie rating prediction trickier than a typical
regression problem, and it's worth understanding for your write-up:

- **Director** and **Actor** are *high-cardinality* categorical features —
  potentially hundreds of unique names, many appearing only once or twice.
  One-hot encoding all of them would create huge, sparse columns the model
  can't learn much from.
- Instead, `feature_engineering.py` uses **target encoding**: each
  director/actor is replaced by the *average rating* of movies they've
  been involved in (e.g. "Director X" → 7.2). This captures real signal —
  some directors/actors are reliably associated with better-rated films.
- **The risk**: if you compute these averages using the whole dataset
  (including the test set), you leak test-set rating information into the
  training features. The model would look great in evaluation but fail on
  truly new movies.
- **The fix used here**: the dataset is split into train/test *first*.
  All target-encoding maps are fit using *only the training set's*
  ratings, then applied to the test set. A director/actor who only shows
  up in the test set (never seen in training) falls back to the overall
  average rating.
- **Smoothing**: a director with only 1 movie in the training set
  shouldn't be trusted as much as one with 50. The encoding for rare
  categories is pulled towards the overall average rating, proportional to
  how few movies they have (see `smoothing` parameter in
  `MovieFeatureEngineer`).
- **Genre** is handled similarly, but a movie can have multiple genres
  (e.g. "Action, Drama"), so each genre is target-encoded individually and
  a movie's genre feature is the average across all its genres.

---

## 6. What each evaluation metric tells you

- **MAE** — average size of prediction errors, in rating points (e.g. MAE
  of 0.5 means predictions are typically off by about half a rating point)
- **MSE / RMSE** — like MAE but penalize larger errors more; RMSE is in the
  same units as Rating, so it's easier to interpret than MSE
- **R² score** — proportion of the variation in ratings the model
  explains (1.0 = perfect; 0 = no better than always predicting the
  average rating). Movie ratings are inherently noisy (even critics
  disagree!), so don't expect R² near 1.0 even with real data — that's
  normal and worth mentioning in your write-up.

---

## 7. Reusing the saved model on new movies

```python
import joblib
import pandas as pd

model = joblib.load("outputs/best_model.pkl")
scaler = joblib.load("outputs/scaler.pkl")
fe = joblib.load("outputs/feature_engineer.pkl")  # the fitted feature engineer

new_movies = pd.DataFrame({
    "Year": [2023],
    "Duration": [118],
    "Genre": ["Action, Sci-Fi"],
    "Votes": [5000],
    "Director": ["James Stone"],     # known directors get learned ratings;
    "Actor 1": ["Alex Hart"],        # unknown ones fall back to the average
    "Actor 2": ["Jordan Bishop"],
    "Actor 3": ["Sam Vance"],
})

X_new = fe.transform(new_movies)
X_new_scaled = scaler.transform(X_new)
predicted_rating = model.predict(X_new_scaled)
print(predicted_rating)
```

---

## 8. Tips for your CodSoft submission

- Push this whole folder to a public GitHub repo.
- For your demo video, `feature_importance.png` is a great talking
  point: walk through which features (director, actor, genre, votes,
  duration, year) matter most, and explain *why* target encoding was
  needed for director/actor instead of one-hot encoding.
- Explicitly mention how you avoided data leakage (fit encoders on
  train only) — this is exactly the kind of detail that shows real
  understanding, not just copy-pasted code.
- Try swapping in a real dataset and compare R² — real movie ratings are
  noisier than this synthetic data, so don't be surprised if R² is lower;
  that's expected and worth discussing.

---

## 9. Troubleshooting

- **"Could not find dataset" error** — run `generate_sample_data.py`
  first, or check the `--data` path you passed.
- **"missing expected column(s)" error** — your CSV must have the exact
  column names listed in Section 2 (case- and space-sensitive, e.g.
  `Actor 1` not `Actor1`). Rename columns in your CSV if needed.
- **Low R² on a real dataset** — this is normal for movie ratings; try
  adding more features if your real dataset has them (e.g. budget,
  language, country, runtime, production company).
# Movie-rating
