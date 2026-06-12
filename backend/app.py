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

BASE     = os.path.dirname(__file__)
PLAYERS  = os.path.join(BASE, "../vct_2025/players_stats/players_stats.csv")
AGENTS   = os.path.join(BASE, "../vct_2025/agents/agents_pick_rates.csv")
MAPSTATS = os.path.join(BASE, "../vct_2025/agents/maps_stats.csv")
SCORES   = os.path.join(BASE, "../vct_2025/matches/maps_scores.csv")
OVERVIEW = os.path.join(BASE, "../vct_2025/matches/overview.csv")
KILLS    = os.path.join(BASE, "../vct_2025/matches/kills_stats.csv")

AGENT_ROLE_MAP = {
    1: ['jett', 'raze', 'neon', 'yoru', 'iso', 'reyna', 'phoenix'],
    2: ['sova', 'fade', 'breach', 'kayo', 'gekko', 'skye', 'tejo'],
    3: ['omen', 'viper', 'astra', 'brimstone', 'harbor', 'clove'],
    4: ['cypher', 'killjoy', 'sage', 'deadlock', 'vyse', 'chamber', 'waylay'],
}
ROLE_NAMES = {1: 'Duelista', 2: 'Iniciador', 3: 'Controlador', 4: 'Centinela'}

PCT_COLS = ["Kill, Assist, Trade, Survive %", "Headshot %"]

PS_FEATURES = [
    "Rating", "Average Combat Score", "Kills Per Round",
    "Average Damage Per Round", "Kill, Assist, Trade, Survive %",
    "First Kills Per Round", "First Deaths Per Round", "Headshot %",
    "Assists Per Round",
]
KI_FEATURES = ["2k", "3k", "4k", "Spike Plants", "Spike Defuses", "1v1", "1v2", "Econ"]
OV_FEATURES = ["rating_attack", "rating_defend", "rating_both",
               "attack_vs_defend", "acs_attack", "acs_defend"]
CALC_FEATURES = ["num_agents", "agent_role"]
PCA_FEATURES  = PS_FEATURES + KI_FEATURES + OV_FEATURES + CALC_FEATURES


def pct_to_float(series):
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return series.str.replace('%', '', regex=False).str.strip().astype(float) / 100.0
    return series


def get_player_role(agents_list):
    counts = {r: 0 for r in AGENT_ROLE_MAP}
    for agent in agents_list:
        al = agent.lower().strip()
        for role_id, role_agents in AGENT_ROLE_MAP.items():
            if al in role_agents:
                counts[role_id] += 1
                break
    dominant = max(counts, key=counts.get)
    return dominant if counts[dominant] > 0 else 4


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


def build_enriched_player_df():
    """One row per player with features from players_stats, kills_stats, and overview."""
    ps = load_players()
    ov = load_overview()
    ki = load_kills()

    # players_stats: pct conversion, multi-agent rows only, average by player
    for col in PCT_COLS:
        if col in ps.columns:
            ps[col] = pct_to_float(ps[col])
    ps_agg = ps[ps["Agents"].str.contains(",", na=False)].copy()
    for col in PS_FEATURES:
        if col in ps_agg.columns:
            ps_agg[col] = pd.to_numeric(ps_agg[col], errors="coerce")
    avail_ps  = [c for c in PS_FEATURES if c in ps_agg.columns]
    ps_player = ps_agg.groupby("Player")[avail_ps].mean().reset_index()

    # kills_stats: average per player
    avail_ki = [c for c in KI_FEATURES if c in ki.columns]
    for col in avail_ki:
        ki[col] = pd.to_numeric(ki[col], errors="coerce")
    ki_player = (ki.groupby("Player")[avail_ki].mean().reset_index()
                 if avail_ki else pd.DataFrame({"Player": ps_player["Player"]}))

    # overview: per-side rating and ACS per player
    for col in PCT_COLS:
        if col in ov.columns:
            ov[col] = pct_to_float(ov[col])
    for col in ["Rating", "Average Combat Score"]:
        if col in ov.columns:
            ov[col] = pd.to_numeric(ov[col], errors="coerce")

    if "Side" in ov.columns and "Rating" in ov.columns:
        def side_avg(side, metric):
            return ov[ov["Side"].str.lower() == side].groupby("Player")[metric].mean()

        ov_df = pd.DataFrame({
            "rating_attack": side_avg("attack", "Rating"),
            "rating_defend": side_avg("defend", "Rating"),
            "rating_both":   side_avg("both",   "Rating"),
        })
        ov_df["attack_vs_defend"] = ov_df["rating_attack"] - ov_df["rating_defend"]
        if "Average Combat Score" in ov.columns:
            ov_df["acs_attack"] = side_avg("attack", "Average Combat Score")
            ov_df["acs_defend"] = side_avg("defend", "Average Combat Score")
        ov_player = ov_df.reset_index()
    else:
        ov_player = pd.DataFrame({"Player": ps_player["Player"]})

    # Calculated fields: num_agents, agent_role, plus metadata
    agent_rows = []
    for player, grp in ps_agg.groupby("Player"):
        all_agents: set = set()
        for ag_str in grp["Agents"].dropna():
            for ag in ag_str.split(","):
                ag = ag.strip()
                if ag:
                    all_agents.add(ag)
        agent_rows.append({
            "Player":     player,
            "num_agents": len(all_agents),
            "agent_role": get_player_role(list(all_agents)),
            "tournament": grp["Tournament"].mode().iloc[0] if len(grp) > 0 else "",
            "team":       grp["Teams"].mode().iloc[0]       if len(grp) > 0 else "",
            "agents":     ", ".join(sorted(all_agents)),
        })
    agent_df = pd.DataFrame(agent_rows)

    merged = (
        ps_player
        .merge(ki_player, on="Player", how="left")
        .merge(ov_player,  on="Player", how="left")
        .merge(agent_df,   on="Player", how="left")
    )
    return merged


def fv(series, key):
    """Safe float extraction from a pandas Series."""
    if key not in series.index:
        return None
    v = series[key]
    return float(v) if pd.notna(v) else None


@app.route("/api/pca")
def pca_endpoint():
    merged = build_enriched_player_df()
    feats  = [f for f in PCA_FEATURES if f in merged.columns]

    X     = merged[feats].copy().apply(pd.to_numeric, errors="coerce")
    imp   = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_imp)

    pca    = PCA(n_components=2)
    coords = pca.fit_transform(X_sc)
    X_df   = pd.DataFrame(X_imp, columns=feats)

    points = []
    for i, row in merged.reset_index(drop=True).iterrows():
        imp_row = X_df.iloc[i]
        role_id = int(row["agent_role"]) if pd.notna(row.get("agent_role")) else 4

        entry = {
            "id":               i,
            "player":           str(row.get("Player", "")),
            "team":             str(row.get("team", "")),
            "tournament":       str(row.get("tournament", "")),
            "agents":           str(row.get("agents", "")),
            "rating":           fv(imp_row, "Rating"),
            "pc1":              float(coords[i, 0]),
            "pc2":              float(coords[i, 1]),
            "agent_role":       role_id,
            "role_name":        ROLE_NAMES.get(role_id, "Centinela"),
            "num_agents":       int(fv(imp_row, "num_agents") or 1),
            "rating_attack":    fv(imp_row, "rating_attack"),
            "rating_defend":    fv(imp_row, "rating_defend"),
            "attack_vs_defend": fv(imp_row, "attack_vs_defend"),
            "avg_acs":          fv(imp_row, "Average Combat Score"),
            "avg_kpr":          fv(imp_row, "Kills Per Round"),
            "avg_adr":          fv(imp_row, "Average Damage Per Round"),
            "avg_kast":         fv(imp_row, "Kill, Assist, Trade, Survive %"),
            "avg_fkr":          fv(imp_row, "First Kills Per Round"),
            "avg_fdr":          fv(imp_row, "First Deaths Per Round"),
            "multi_kills_2k":   fv(imp_row, "2k"),
            "multi_kills_3k":   fv(imp_row, "3k"),
            "spike_plants":     fv(imp_row, "Spike Plants"),
            "spike_defuses":    fv(imp_row, "Spike Defuses"),
        }
        for feat in feats:
            entry[feat] = fv(imp_row, feat)
        points.append(entry)

    ev = pca.explained_variance_ratio_.tolist()
    return jsonify({
        "points":                   points,
        "explained_variance":       ev,
        "total_variance_explained": float(sum(ev)),
    })


@app.route("/api/pca/loadings")
def pca_loadings():
    merged = build_enriched_player_df()
    feats  = [f for f in PCA_FEATURES if f in merged.columns]

    X    = merged[feats].copy().apply(pd.to_numeric, errors="coerce")
    imp  = SimpleImputer(strategy="median")
    sc   = StandardScaler()
    X_sc = sc.fit_transform(imp.fit_transform(X))

    pca = PCA(n_components=2)
    pca.fit(X_sc)

    comps  = pca.components_
    result = []
    for j, feat in enumerate(feats):
        p1, p2 = float(comps[0, j]), float(comps[1, j])
        result.append({"feature": feat, "pc1": p1, "pc2": p2,
                        "importance": float(np.sqrt(p1**2 + p2**2))})
    result.sort(key=lambda x: x["importance"], reverse=True)
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

    agg_rows = p_rows[p_rows["Agents"].str.contains(",", na=False)].copy()

    num_cols = [
        "Rating", "Average Combat Score", "Kills Per Round",
        "Average Damage Per Round", "Kill, Assist, Trade, Survive %",
        "First Kills Per Round", "First Deaths Per Round", "Headshot %",
    ]
    for col in num_cols:
        if col in agg_rows.columns:
            agg_rows[col] = pd.to_numeric(agg_rows[col], errors="coerce")

    all_agg = df[df["Agents"].str.contains(",", na=False)].copy()
    for col in PCT_COLS:
        if col in all_agg.columns:
            all_agg[col] = pct_to_float(all_agg[col])
    for col in num_cols:
        if col in all_agg.columns:
            all_agg[col] = pd.to_numeric(all_agg[col], errors="coerce")
        if col in agg_rows.columns and agg_rows[col].isna().any():
            agg_rows[col] = agg_rows[col].fillna(all_agg[col].median())

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

    all_agents: set = set()
    for ag_str in p_rows["Agents"].dropna():
        for ag in ag_str.split(","):
            ag = ag.strip()
            if ag:
                all_agents.add(ag)
    agents = sorted(all_agents)

    k_rows    = ki[ki["Player"].str.lower() == player_name.lower()].copy()
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
        "player":            p_rows["Player"].iloc[0],
        "teams":             teams,
        "tournaments":       tournaments,
        "agents":            agents,
        "avg_rating":        safe_mean("Rating"),
        "max_rating":        safe_max("Rating"),
        "min_rating":        safe_min("Rating"),
        "avg_acs":           safe_mean("Average Combat Score"),
        "avg_kpr":           safe_mean("Kills Per Round"),
        "avg_adr":           safe_mean("Average Damage Per Round"),
        "avg_kast":          safe_mean("Kill, Assist, Trade, Survive %"),
        "avg_fkr":           safe_mean("First Kills Per Round"),
        "avg_fdr":           safe_mean("First Deaths Per Round"),
        "avg_headshot":      safe_mean("Headshot %"),
        "total_tournaments": len(tournaments),
        "kills_stats":       kills_agg,
    })


@app.route("/api/compare", methods=["POST"])
def compare():
    body  = request.get_json(force=True)
    names = [n.lower() for n in body.get("players", [])]

    df  = load_players()
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

    sel     = agg[agg["Player"].str.lower().isin(names)].copy()
    grouped = sel.groupby("Player")[COMPARE_COLS].mean().reset_index()
    meta    = sel.groupby("Player").agg(
        Teams=("Teams", "first"), Tournament=("Tournament", "first")
    ).reset_index()
    grouped = grouped.merge(meta, on="Player", how="left")

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
            entry[col]           = float(row[col])           if not pd.isna(row[col])           else None
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
        .mean().reset_index()
        .rename(columns={"Pick Rate": "avg_pick_rate"})
        .sort_values("avg_pick_rate", ascending=False)
        .to_dict(orient="records")
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
