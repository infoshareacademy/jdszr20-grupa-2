# recommend.py
# -*- coding: utf-8 -*-
"""
Moduł rekomendacji dla Plantelligence.
Strategia: (1) KMeans → wybór klastra, (2) cosine similarity w obrębie klastra.

Wymagane artefakty w ./artifacts:
- ohe.pkl
- scaler.pkl
- pca.pkl
- kmeans.pkl
- plant_embeddings.npy
- feature_list.json
oraz plik danych wejściowych (po feature engineeringu), np.:
- house_plants_features.csv  (zawiera m.in. kolumny: latin + FEATURE_COLS)
"""

from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


# ---- ŚCIEŻKI ----
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

# Domyślna nazwa pliku z cechami (zmień, jeśli u Ciebie nazywa się inaczej)
DEFAULT_FEATURE_CSV = HERE / "data" / "plants_clean_final.csv"





# ---- ŁADOWANIE ARTEFAKTÓW ----
def _load_artifacts(
    feature_csv: Path | str = DEFAULT_FEATURE_CSV
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, object, object, object, List[str]]:
    """Ładuje dane i modele. Zwraca:
    df, embeddings(PCA), labels(KMeans), ohe, scaler, pca, feature_list
    """
    feature_csv = Path(feature_csv)

    if not feature_csv.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku z cechami: {feature_csv}\n"
            f"Ustaw poprawną nazwę w DEFAULT_FEATURE_CSV albo przekaż ścieżkę w parametrze."
        )

    df = pd.read_csv(feature_csv)

    # artefakty
    ohe = joblib.load(ARTIFACTS / "ohe.pkl")
    scaler = joblib.load(ARTIFACTS / "scaler.pkl")
    pca = joblib.load(ARTIFACTS / "pca.pkl")
    kmeans = joblib.load(ARTIFACTS / "kmeans.pkl")

    # --- poprawione ładowanie feature_list ---
    with open(ARTIFACTS / "feature_list.json", "r", encoding="latin-1") as f:
        feature_list = json.load(f)

    if isinstance(feature_list, dict) and "features_order" in feature_list:
        feature_list = [item["name"] for item in feature_list["features_order"]]
    # --- koniec poprawki ---

    embeddings = np.load(ARTIFACTS / "plant_embeddings.npy")

    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Niezgodność liczby wierszy: df={len(df)} vs embeddings={embeddings.shape[0]}."
        )

    labels = kmeans.labels_
    if labels is None or len(labels) != len(df):
        labels = kmeans.predict(embeddings)

    return df, embeddings, labels, ohe, scaler, pca, feature_list


# ---- POMOCNICZE ----
def _to_embedding_from_features(
    features: Dict[str, object],
    ohe,
    scaler,
    pca,
    feature_list: List[str]
) -> np.ndarray:
    """
    Buduje wektor cech w takiej samej kolejności jak w treningu:
    [OHE(kategoryczne)] + [Scaler(2 numeryczne)] + [binarka (bez skalowania)]
    i przekształca go do embeddingu PCA (shape: 1 x d).
    """
    import json

    # 1) Jednowierszowy DF w kolejności feature_list
    row = {col: features.get(col, np.nan) for col in feature_list}
    X = pd.DataFrame([row], columns=feature_list)

    # 2) Wczytaj metadane, żeby znać podziały kolumn
    with open(ARTIFACTS / "feature_list.json", "r", encoding="latin-1") as f:
        fl_meta = json.load(f)

    # Jeśli plik ma strukturę z 'features_order', wyciągamy typy
    if isinstance(fl_meta, dict) and "features_order" in fl_meta:
        feat_order = fl_meta["features_order"]
        cat_cols = [x["name"] for x in feat_order if x.get("type") == "categorical"]
        num_all  = [x["name"] for x in feat_order if x.get("type") == "numeric"]
    else:
        # awaryjnie: zakładamy brak metadanych (nie powinno się zdarzyć)
        cat_cols, num_all = [], []

    # Z Twojej specyfikacji: 2 numeryczne do skalowania, 'toxicity_any' to binarka bez skalowania
    num_scaled = [c for c in num_all if c != "toxicity_any"]
    bin_cols   = ["toxicity_any"] if "toxicity_any" in num_all else []

    # 3) Transformacje: OHE dla kategorycznych
    X_cat = X[cat_cols] if cat_cols else pd.DataFrame(index=X.index)
    X_ohe = ohe.transform(X_cat)
    # zbij do numpy jeśli sparse
    if hasattr(X_ohe, "toarray"):
        X_ohe = X_ohe.toarray()

    # 4) Scaler dla dwóch numerycznych
    X_num = X[num_scaled] if num_scaled else pd.DataFrame(index=X.index)
    X_scaled = scaler.transform(X_num) if len(num_scaled) > 0 else np.empty((len(X), 0))

    # 5) Binarka bez skalowania
    X_bin = X[bin_cols].to_numpy(dtype=float) if len(bin_cols) > 0 else np.empty((len(X), 0))

    # 6) Sklejenie w tej samej kolejności: OHE | SCALED | BIN
    X_full = np.hstack([X_ohe, X_scaled, X_bin])

    # 7) PCA → embedding (1 x d)
    X_emb = pca.transform(X_full)
    return X_emb


def _topn_in_cluster(
    target_vec: np.ndarray,
    cluster_id: int,
    embeddings: np.ndarray,
    labels: np.ndarray,
    latin: pd.Series,
    top_n: int = 5,
    exclude_index: int | None = None
) -> pd.DataFrame:
    """Zwraca top-N najbardziej podobnych w obrębie klastra."""
    idx = np.where(labels == cluster_id)[0]
    if idx.size == 0:
        return pd.DataFrame(columns=["latin", "similarity", "rank"])

    # macierz embeddingów danego klastra
    E = embeddings[idx]  # (m, d)

    sims = cosine_similarity(target_vec.reshape(1, -1), E).ravel()  # (m,)
    # mapowanie do indeksów globalnych
    sim_df = pd.DataFrame({
        "global_idx": idx,
        "latin": latin.iloc[idx].values,
        "similarity": sims
    })

    # opcjonalnie wyklucz roślinę źródłową
    if exclude_index is not None:
        sim_df = sim_df[sim_df["global_idx"] != exclude_index]

    sim_df = sim_df.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
    sim_df.insert(0, "rank", range(1, len(sim_df) + 1))
    return sim_df[["rank", "latin", "similarity"]]


# ---- INTERFEJS PUBLICZNY ----
def recommend_by_plant(
    latin_name: str,
    top_n: int = 5,
    feature_csv: Path | str = DEFAULT_FEATURE_CSV
) -> pd.DataFrame:
    """Rekomendacje podobnych roślin do podanej (po nazwie łacińskiej).
    1) Bierzemy embedding rośliny,
    2) wybieramy jej klaster,
    3) liczymy cosine similarity w tym klastrze,
    4) zwracamy top-N.
    """
    df, embeddings, labels, ohe, scaler, pca, feature_list = _load_artifacts(feature_csv)
    if "latin" not in df.columns:
        raise KeyError("W pliku z danymi brakuje kolumny 'latin'.")

    hits = df.index[df["latin"] == latin_name].tolist()
    if not hits:
        raise ValueError(f"Nie znaleziono rośliny o nazwie łacińskiej: {latin_name}")

    idx = hits[0]
    cluster_id = labels[idx]
    target_vec = embeddings[idx]

    return _topn_in_cluster(
        target_vec=target_vec,
        cluster_id=cluster_id,
        embeddings=embeddings,
        labels=labels,
        latin=df["latin"],
        top_n=top_n,
        exclude_index=idx
    )


def recommend_by_answers(
    answers: Dict[str, object],
    top_n: int = 5,
    feature_csv: Path | str = DEFAULT_FEATURE_CSV
) -> pd.DataFrame:
    """Rekomendacje na podstawie odpowiedzi z quizu (słownik cech).
    Oczekuje kluczy zgodnych z feature_list.json.
    Przykład:
        answers = {
            "light": "Low light",
            "watering_group": "Moderate",
            "temp_band": "Cool-tolerant",
            "category_group": "Paproć / roślina zielona",
            "toxicity_any": 0,
            "heat_band_simplified": "Not heat-tolerant"
        }
    """
    df, embeddings, labels, ohe, scaler, pca, feature_list = _load_artifacts(feature_csv)

    # embedding użytkownika
    user_emb = _to_embedding_from_features(answers, ohe, scaler, pca, feature_list)  # (1, d)

    # wybór klastra użytkownika (w przestrzeni embeddingów)
    # Uwaga: większość implementacji KMeans przewiduje w oryginalnej przestrzeni po-skalowaniu;
    # my pracujemy w PCA, więc konsekwentnie predykcja w tej samej przestrzeni co embeddings.
    kmeans = joblib.load(ARTIFACTS / "kmeans.pkl")
    cluster_id = int(kmeans.predict(user_emb)[0])

    # top-N wewnątrz klastra
    recs = _topn_in_cluster(
        target_vec=user_emb.ravel(),
        cluster_id=cluster_id,
        embeddings=embeddings,
        labels=labels,
        latin=df["latin"],
        top_n=top_n,
        exclude_index=None
    )
    return recs


# ---- DEMO / CLI ----
if __name__ == "__main__":
    # Przykład 1: podobne do znanej rośliny
    try:
        demo = recommend_by_plant("Ficus elastica", top_n=5)
        print("\nPodobne do 'Ficus elastica':")
        print(demo.to_string(index=False))
    except Exception as e:
        print("Demo (by_plant) pominiete:", e)

    # Przykład 2: podobne do odpowiedzi z quizu
    try:
        sample_answers = {
            "light": "Low light",
            "watering_group": "Moderate",
            "temp_band": "Cool-tolerant",
            "category_group": "Paproć / roślina zielona",
            "toxicity_any": 0,
            "heat_band_simplified": "Not heat-tolerant"
        }
        demo2 = recommend_by_answers(sample_answers, top_n=5)
        print("\nRekomendacje dla przykładowych odpowiedzi:")
        print(demo2.to_string(index=False))
    except Exception as e:
        print("Demo (by_answers) pominiete:", e)
