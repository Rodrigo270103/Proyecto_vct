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
    ki = load_kills()

    p_rows = df[df["Player"].str.lower() == player_name.lower()].copy()
    if p_rows.empty:
        return jsonify({"error": "Player not found"}), 404

    for col in PCT_COLS:
        if col in p_rows.columns:
            p_rows[col] = pct_to_float(p_rows[col])

    # Only aggregated rows (multi-agent) represent full-tournament stats
    agg_rows = p_rows[p_rows["Agents"].str.contains(",", na=False)].copy()

    num_cols = [
        "Rating", "Average Combat Score", "Kills Per Round",
        "Average Damage Per Round", "Kill, Assist, Trade, Survive %",
        "First Kills Per Round", "First Deaths Per Round", "Headshot %",
    ]
    for col in num_cols:
        if col in agg_rows.columns:
            agg_rows[col] = pd.to_numeric(agg_rows[col], errors="coerce")

    # Impute NaN with global median so players with sparse data still get a value
    all_agg = df[df["Agents"].str.contains(",", na=False)].copy()
    for col in PCT_COLS:
        if col in all_agg.columns:
            all_agg[col] = pct_to_float(all_agg[col])
    for col in num_cols:
        if col in all_agg.columns:
            all_agg[col] = pd.to_numeric(all_agg[col], errors="coerce")
        if col in agg_rows.columns and agg_rows[col].isna().any():
            global_median = all_agg[col].median()
            agg_rows[col] = agg_rows[col].fillna(global_median)

    def safe_mean(col):
        if col not in agg_rows.columns or agg_rows[col].isna().all():
            return 0.0
        return float(agg_rows[col].mean())

    def safe_max(col):
        if col not in agg_rows.columns or agg_rows[col].isna().all():
            return 0.0
        return float(agg_rows[col].max())

    def safe_min(col):
        if col not in agg_rows.columns or agg_rows[col].isna().all():
            return 0.0
        return float(agg_rows[col].min())

    teams       = sorted(p_rows["Teams"].dropna().unique().tolist())
    tournaments = sorted(p_rows["Tournament"].dropna().unique().tolist())

    all_agents = set()
    for agents_str in p_rows["Agents"].dropna():
        for a in agents_str.split(","):
            a = a.strip()
            if a:
                all_agents.add(a)
    agents = sorted(all_agents)

    k_rows = ki[ki["Player"].str.lower() == player_name.lower()].copy()
    kills_agg = None
    if not k_rows.empty:
        for col in ["2k", "3k", "4k", "5k"]:
            if col in k_rows.columns:
                k_rows[col] = pd.to_numeric(k_rows[col], errors="coerce").fillna(0)
        kills_agg = {
            "2k": int(k_rows["2k"].sum()) if "2k" in k_rows.columns else 0,
            "3k": int(k_rows["3k"].sum()) if "3k" in k_rows.columns else 0,
            "4k": int(k_rows["4k"].sum()) if "4k" in k_rows.columns else 0,
            "5k": int(k_rows["5k"].sum()) if "5k" in k_rows.columns else 0,
        }

    return jsonify({
        "player":           p_rows["Player"].iloc[0],
        "teams":            teams,
        "tournaments":      tournaments,
        "agents":           agents,
        "avg_rating":       safe_mean("Rating"),
        "max_rating":       safe_max("Rating"),
        "min_rating":       safe_min("Rating"),
        "avg_acs":          safe_mean("Average Combat Score"),
        "avg_kpr":          safe_mean("Kills Per Round"),
        "avg_adr":          safe_mean("Average Damage Per Round"),
        "avg_kast":         safe_mean("Kill, Assist, Trade, Survive %"),
        "avg_fkr":          safe_mean("First Kills Per Round"),
        "avg_fdr":          safe_mean("First Deaths Per Round"),
        "avg_headshot":     safe_mean("Headshot %"),
        "total_tournaments": len(tournaments),
        "kills_stats":      kills_agg,
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

    for col in COMPARE_COLS:
        agg[col] = pd.to_numeric(agg[col], errors="coerce")

    selected_all = agg[agg["Player"].str.lower().isin(names)].copy()

    # One row per player: average all numeric metrics across their records
    grouped = selected_all.groupby("Player")[COMPARE_COLS].mean().reset_index()

    # Attach first-seen team/tournament metadata
    meta = (selected_all.groupby("Player")
            .agg(Teams=("Teams", "first"), Tournament=("Tournament", "first"))
            .reset_index())
    grouped = grouped.merge(meta, on="Player", how="left")

    # Normalize 0-1 relative to the full player pool
    for col in COMPARE_COLS:
        mn, mx = agg[col].min(), agg[col].max()
        if mx != mn:
            grouped[col + "_norm"] = ((grouped[col] - mn) / (mx - mn)).clip(0, 1)
        else:
            grouped[col + "_norm"] = 0.5

    result = []
    for _, row in grouped.iterrows():
        entry = {
            "player":     row["Player"],
            "team":       row.get("Teams", ""),
            "tournament": row.get("Tournament", ""),
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
