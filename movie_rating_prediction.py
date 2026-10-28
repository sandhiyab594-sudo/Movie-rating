"""
movie_rating_prediction.py
-----------------------------
End-to-end movie rating prediction pipeline.

Steps:
  1. Load & explore the movie data
  2. Split into train/test sets (BEFORE any target-encoding, to avoid leakage)
  3. Feature engineering: target-encode Director / Actors / Genre,
     using only the training set's ratings (see feature_engineering.py)
  4. Scale numeric features
  5. Train regression models: Linear Regression and Random Forest Regressor
  6. Evaluate with MAE, MSE, RMSE, R2
  7. Visualize: rating distribution, avg rating by genre, actual vs
     predicted, residuals, feature importance
  8. Save the best model + the fitted feature engineer

USAGE
-----
    python movie_rating_prediction.py
    python movie_rating_prediction.py --data data/movies.csv

See README.md for full setup + explanation.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import MovieFeatureEngineer, GENRE_LIST_COL, TARGET_COL

OUTPUT_DIR = "outputs"
REQUIRED_COLS = ["Year", "Duration", "Genre", "Votes", "Director",
                  "Actor 1", "Actor 2", "Actor 3", "Rating"]


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'.\n"
            f"  -> Run 'python generate_sample_data.py' first to create a "
            f"synthetic demo dataset at data/movies_sample.csv, OR\n"
            f"  -> place your own dataset at that path (see README for the "
            f"required column names)."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected column(s): {missing}.\n"
            f"Expected columns: {REQUIRED_COLS}"
        )
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# --------------------------------------------------------------------------
# 2. EXPLORE
# --------------------------------------------------------------------------
def explore_data(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nRating distribution:")
    print(df[TARGET_COL].describe())

    # Rating distribution histogram
    plt.figure(figsize=(5.5, 4))
    sns.histplot(df[TARGET_COL], bins=25, kde=True, color="#4C72B0")
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rating_distribution.png"), dpi=150)
    plt.close()

    # Average rating by genre (a movie can have multiple genres -> explode)
    exploded = df.assign(
        Genre=df[GENRE_LIST_COL].str.split(",")
    ).explode(GENRE_LIST_COL)
    exploded[GENRE_LIST_COL] = exploded[GENRE_LIST_COL].str.strip()
    genre_avg = exploded.groupby(GENRE_LIST_COL)[TARGET_COL].mean().sort_values()

    plt.figure(figsize=(6, 5))
    genre_avg.plot(kind="barh", color="#55A868")
    plt.title("Average Rating by Genre")
    plt.xlabel("Average Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "avg_rating_by_genre.png"), dpi=150)
    plt.close()

    # Correlation heatmap for numeric features
    numeric_df = df[["Year", "Duration", "Votes", TARGET_COL]].copy()
    numeric_df["log_votes"] = np.log1p(numeric_df["Votes"])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (numeric features)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    print("Saved: outputs/rating_distribution.png, outputs/avg_rating_by_genre.png, "
          "outputs/correlation_heatmap.png")


# --------------------------------------------------------------------------
# 3-4. SPLIT + FEATURE ENGINEERING (encoders fit on TRAIN ONLY)
# --------------------------------------------------------------------------
def split_and_engineer_features(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"\nTrain: {train_df.shape[0]} rows | Test: {test_df.shape[0]} rows")

    fe = MovieFeatureEngineer()
    X_train_raw = fe.fit_transform(train_df)     # fit + transform on TRAIN only
    X_test_raw = fe.transform(test_df)            # transform-only on TEST (no leakage)

    y_train = train_df[TARGET_COL].reset_index(drop=True)
    y_test = test_df[TARGET_COL].reset_index(drop=True)
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)

    return X_train, X_test, y_train, y_test, fe, scaler


# --------------------------------------------------------------------------
# 5. TRAIN MODELS
# --------------------------------------------------------------------------
def train_models(X_train, y_train):
    models = {}

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    models["Linear Regression"] = lin_reg

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    return models


# --------------------------------------------------------------------------
# 6. EVALUATE
# --------------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n===== {name} =====")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    plot_actual_vs_predicted(y_test, y_pred, name)
    plot_residuals(y_test, y_pred, name)

    return {"name": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def plot_actual_vs_predicted(y_test, y_pred, name):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "--", color="gray", label="Perfect prediction")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"Actual vs Predicted — {name}")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"actual_vs_predicted_{name.replace(' ', '_').lower()}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_residuals(y_test, y_pred, name):
    residuals = y_test - y_pred
    plt.figure(figsize=(5, 4))
    plt.scatter(y_pred, residuals, alpha=0.6, color="#C44E52")
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Predicted Rating")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residual Plot — {name}")
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"residuals_{name.replace(' ', '_').lower()}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_feature_importance(models, feature_names):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    lin_reg = models["Linear Regression"]
    coefs = pd.Series(lin_reg.coef_, index=feature_names).sort_values()
    coefs.plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Linear Regression Coefficients\n(scaled features)")
    axes[0].set_xlabel("Coefficient value")

    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
    importances.plot(kind="barh", ax=axes[1], color="#55A868")
    axes[1].set_title("Random Forest Feature Importance")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Movie Rating Prediction Pipeline")
    parser.add_argument(
        "--data", default="data/movies_sample.csv",
        help="Path to the movies CSV (default: data/movies_sample.csv). "
             "See README.md for the required columns."
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    df = load_data(args.data)

    print("\n" + "=" * 60)
    print("STEP 2: Exploring data")
    print("=" * 60)
    explore_data(df)

    print("\n" + "=" * 60)
    print("STEP 3: Train/test split + feature engineering")
    print("(Director/Actor/Genre target-encoders are fit on the TRAINING")
    print(" set only, then applied to the test set, to avoid data leakage)")
    print("=" * 60)
    X_train, X_test, y_train, y_test, fe, scaler = split_and_engineer_features(
        df, test_size=args.test_size
    )
    print(f"Engineered features: {fe.feature_names_}")

    print("\n" + "=" * 60)
    print("STEP 4: Training models")
    print("=" * 60)
    models = train_models(X_train, y_train)

    print("\n" + "=" * 60)
    print("STEP 5: Evaluating models")
    print("=" * 60)
    results = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X_test, y_test))

    plot_feature_importance(models, fe.feature_names_)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(results)[["name", "MAE", "MSE", "RMSE", "R2"]]
    summary_df.columns = ["Model", "MAE", "MSE", "RMSE", "R2 Score"]
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    best_idx = summary_df["R2 Score"].idxmax()
    best_name = summary_df.loc[best_idx, "Model"]
    best_model = models[best_name]
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(fe, os.path.join(OUTPUT_DIR, "feature_engineer.pkl"))
    print(f"\nBest model by R2 score: {best_name} -> saved to outputs/best_model.pkl")
    print("All plots and results saved in the 'outputs/' folder.")


if __name__ == "__main__":
    main()
