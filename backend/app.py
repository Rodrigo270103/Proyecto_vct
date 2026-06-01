from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import os

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(__file__)
PLAYERS   = os.path.join(BASE, "../vct_2025/players_stats/players_stats.csv")
AGENTS    = os.path.join(BASE, "../vct_2025/agents/agents_pick_rates.csv")
MAPSTATS  = os.path.join(BASE, "../vct_2025/agents/maps_stats.csv")
SCORES    = os.path.join(BASE, "../vct_2025/matches/maps_scores.csv")
OVERVIEW  = os.path.join(BASE, "../vct_2025/matches/overview.csv")
KILLS     = os.path.join(BASE, "../vct_2025/matches/kills_stats.csv")

def pct_to_float(series):
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return series.str.replace('%', '', regex=False).str.strip().astype(float) / 100.0
    return series

def load_players():
    df = pd.read_csv(PLAYERS)
    df.columns = df.columns.str.strip()
    return df

def load_overview():
    df = pd.read_csv(OVERVIEW)
    df.columns = df.columns.str.strip()
    return df

def load_kills():
    df = pd.read_csv(KILLS)
    df.columns = df.columns.str.strip()
    return df

PCA_FEATURES = [
    "Rating",
    "Average Combat Score",
    "Kills Per Round",
    "Kill, Assist, Trade, Survive %",
    "Average Damage Per Round",
    "Assists Per Round",
    "First Kills Per Round",
    "First Deaths Per Round",
    "Headshot %",
]

PCT_COLS = ["Kill, Assist, Trade, Survive %", "Headshot %"]

@app.route("/api/pca")
def pca_endpoint():
    df = load_players()
    agg = df[df["Agents"].str.contains(",", na=False)].copy()

    for col in PCT_COLS:
        if col in agg.columns:
            agg[col] = pct_to_float(agg[col])

    X = agg[PCA_FEATURES].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    # Replace X with imputed DataFrame for downstream metric retrieval
    X = pd.DataFrame(X_imp, columns=PCA_FEATURES)

    result = []
    for i, row in agg.reset_index(drop=True).iterrows():
        entry = {
            "id": i,
            "player": row.get("Player", ""),
            "team": row.get("Teams", ""),
            "tournament": row.get("Tournament", ""),
            "agents": row.get("Agents", ""),
            "rating": float(X.iloc[i]["Rating"]) if not pd.isna(X.iloc[i]["Rating"]) else None,
            "acs": float(X.iloc[i]["Average Combat Score"]) if not pd.isna(X.iloc[i]["Average Combat Score"]) else None,
            "pc1": float(coords[i, 0]),
            "pc2": float(coords[i, 1]),
        }
        for feat in PCA_FEATURES:
            val = X.iloc[i][feat]
            entry[feat] = float(val) if not pd.isna(val) else None
        result.append(entry)

    return jsonify(result)


@app.route("/api/pca/loadings")
def pca_loadings():
    df = load_players()
    agg = df[df["Agents"].str.contains(",", na=False)].copy()

    for col in PCT_COLS:
        if col in agg.columns:
            agg[col] = pct_to_float(agg[col])

    X = agg[PCA_FEATURES].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    loadings = pca.components_
    result = []
    for j, feat in enumerate(PCA_FEATURES):
        result.append({"feature": feat, "pc1": float(loadings[0, j]), "pc2": float(loadings[1, j])})

    return jsonify(result)


@app.route("/api/player/<player_name>")
def player_detail(player_name):
    df = load_players()
    ov = load_overview()
    ki = load_kills()

    p_rows = df[df["Player"].str.lower() == player_name.lower()]
    o_rows = ov[ov["Player"].str.lower() == player_name.lower()]
    k_rows = ki[ki["Player"].str.lower() == player_name.lower()]

    def clean(frame):
        return frame.where(pd.notnull(frame), None).to_dict(orient="records")

    return jsonify({
        "players_stats": clean(p_rows),
        "overview": clean(o_rows),
        "kills_stats": clean(k_rows),
    })


@app.route("/api/compare", methods=["POST"])
def compare():
    body = request.get_json(force=True)
    names = [n.lower() for n in body.get("players", [])]

    df = load_players()
    agg = df[df["Agents"].str.contains(",", na=False)].copy()

    for col in PCT_COLS:
        if col in agg.columns:
            agg[col] = pct_to_float(agg[col])

    COMPARE_COLS = [
        "Rating", "Average Combat Score", "Kills Per Round",
        "Average Damage Per Round", "Kill, Assist, Trade, Survive %",
        "First Kills Per Round",
    ]

    selected = agg[agg["Player"].str.lower().isin(names)].copy()
    for col in COMPARE_COLS:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")

    # Normalize 0-1 across all agg players for radar
    for col in COMPARE_COLS:
        agg[col] = pd.to_numeric(agg[col], errors="coerce")
        mn, mx = agg[col].min(), agg[col].max()
        if mx != mn:
            selected[col + "_norm"] = (selected[col] - mn) / (mx - mn)
        else:
            selected[col + "_norm"] = 0.5

    result = []
    for _, row in selected.iterrows():
        entry = {
            "player": row["Player"],
            "team": row.get("Teams", ""),
            "tournament": row.get("Tournament", ""),
            "agents": row.get("Agents", ""),
        }
        for col in COMPARE_COLS:
            entry[col] = float(row[col]) if not pd.isna(row[col]) else None
            entry[col + "_norm"] = float(row[col + "_norm"]) if not pd.isna(row[col + "_norm"]) else 0
        result.append(entry)

    return jsonify(result)


@app.route("/api/agents")
def agents_endpoint():
    df = pd.read_csv(AGENTS)
    df.columns = df.columns.str.strip()
    df["Pick Rate"] = pct_to_float(df["Pick Rate"])
    result = (
        df.groupby("Agent")["Pick Rate"]
        .mean()
        .reset_index()
        .rename(columns={"Pick Rate": "avg_pick_rate"})
        .sort_values("avg_pick_rate", ascending=False)
        .to_dict(orient="records")
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
