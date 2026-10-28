

import numpy as np
import pandas as pd

GENRE_LIST_COL = "Genre"
DIRECTOR_COL = "Director"
ACTOR_COLS = ["Actor 1", "Actor 2", "Actor 3"]
NUMERIC_COLS = ["Year", "Duration", "Votes"]
TARGET_COL = "Rating"


class MovieFeatureEngineer:
    """
    Fits all encoders on a training DataFrame, then transforms any
    DataFrame (train or test) into a numeric feature matrix.
    """

    def __init__(self, smoothing=8):
        self.smoothing = smoothing
        self.global_mean_ = None
        self.director_map_ = None
        self.actor_map_ = None      # one shared mapping, built from all 3 actor columns
        self.genre_map_ = None      # per-genre average rating
        self.feature_names_ = None

    # ---- internal helper: smoothed target-encoding map for one column ----
    def _fit_target_map(self, values, targets):
        df = pd.DataFrame({"cat": values, "target": targets})
        stats = df.groupby("cat")["target"].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
            stats["count"] + self.smoothing
        )
        return smoothed.to_dict()

    def fit(self, df):
        df = df.copy()
        self.global_mean_ = df[TARGET_COL].mean()

        # --- Director: simple single-category target encoding ---
        self.director_map_ = self._fit_target_map(df[DIRECTOR_COL], df[TARGET_COL])

        # --- Actors: pool ALL appearances across Actor 1/2/3 into one mapping ---
        actor_values, actor_targets = [], []
        for col in ACTOR_COLS:
            actor_values.extend(df[col].tolist())
            actor_targets.extend(df[TARGET_COL].tolist())
        self.actor_map_ = self._fit_target_map(pd.Series(actor_values), pd.Series(actor_targets))

        # --- Genre: a movie can have multiple genres ("Action, Drama") ---
        # explode into one row per genre, then target-encode each genre
        exploded_genre, exploded_target = [], []
        for genres, target in zip(df[GENRE_LIST_COL], df[TARGET_COL]):
            for g in [g.strip() for g in str(genres).split(",")]:
                exploded_genre.append(g)
                exploded_target.append(target)
        self.genre_map_ = self._fit_target_map(pd.Series(exploded_genre), pd.Series(exploded_target))

        self.feature_names_ = [
            "director_encoded", "actor_encoded", "genre_encoded", "num_genres",
        ] + NUMERIC_COLS + ["log_votes"]

        return self

    def transform(self, df):
        df = df.copy()

        director_encoded = df[DIRECTOR_COL].map(self.director_map_).fillna(self.global_mean_)

        actor_encoded = pd.concat(
            [df[col].map(self.actor_map_).fillna(self.global_mean_) for col in ACTOR_COLS],
            axis=1,
        ).mean(axis=1)

        def encode_genres(genre_str):
            genres = [g.strip() for g in str(genre_str).split(",")]
            scores = [self.genre_map_.get(g, self.global_mean_) for g in genres]
            return np.mean(scores)

        genre_encoded = df[GENRE_LIST_COL].apply(encode_genres)
        num_genres = df[GENRE_LIST_COL].apply(lambda s: len(str(s).split(",")))

        features = pd.DataFrame({
            "director_encoded": director_encoded,
            "actor_encoded": actor_encoded,
            "genre_encoded": genre_encoded,
            "num_genres": num_genres,
        }, index=df.index)

        for col in NUMERIC_COLS:
            features[col] = df[col]
        features["log_votes"] = np.log1p(df["Votes"])

        return features[self.feature_names_]

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
