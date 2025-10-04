## 📂 Project Structure

Struktura repozytorium Plantelligence:

project ML/
│
├── data/ # surowe i przetworzone dane
│ └── plants_clean_final.csv
│
├── artifacts/ # zapisane artefakty feature transformation
│ ├── ohe.pkl # OneHotEncoder (18 kolumn)
│ ├── feature_names_ohe.json # nazwy kolumn po OHE
│ ├── scaler.pkl # StandardScaler (2 kolumny)
│ ├── pca.pkl # PCA (10 wymiarów)
│ ├── plant_embeddings.npy # embeddingi wszystkich roślin (159 × 10)
│ └── plant_embeddings.csv # embeddingi z nazwami (do debugowania)
│
├── notebooks/ # Jupyter notebooki z analizą i pipeline
│ └── feature_transformation.ipynb
│
├── inference_utils.py # funkcje inference (build_feature_matrix, to_embedding)
├── similarity.py # funkcje wyszukiwania podobnych roślin (cosine similarity)
└── README.md # dokumentacja projektu


### Opis
- **data/** → dane wejściowe do modelu (csv, inne źródła).  
- **artifacts/** → wszystkie artefakty zapisane podczas Feature Transformation (ohe, scaler, pca, embeddingi).  
- **notebooks/** → analizy krok po kroku (EDA, feature engineering, transformacje).  
- **inference_utils.py** → funkcje do przygotowania wektorów cech i embeddingów.  
- **similarity.py** → logika top-N rekomendacji (cosine similarity).  
- **README.md** → dokumentacja repo.


## 🚀 Usage (Feature Transformation + Inference)

Ten projekt buduje stabilny wektor cech dla roślin i zamienia go w embedding PCA (10D), który można wykorzystać do rekomendacji.

### 1. Budowa wektora cech (21 kolumn)
OHE (18 kolumn) + Scaler (2 kolumny) + Binarka (1 kolumna).
```python
import pandas as pd
from inference_utils import build_feature_matrix

df = pd.read_csv("data/plants_clean_final.csv")
X_full, feature_names = build_feature_matrix(df)
print(X_full.shape)   # (n, 21)

Embedding przez PCA (10D)

from inference_utils import to_embedding

X_pca, feature_names = to_embedding(df)
print(X_pca.shape)    # (n, 10)

Cosine similarity (top-N podobnych roślin)

from similarity import top_n_similar

print(top_n_similar("Aeschynanthus lobianus", n=5))

Artefakty (artefacts/)

ohe.pkl – zakodowany OneHotEncoder

feature_names_ohe.json – nazwy 18 kolumn po OHE

scaler.pkl – StandardScaler (dla cold/heat tolerant)

pca.pkl – PCA (10 wymiarów)

plant_embeddings.npy – embeddingi wszystkich roślin (159 × 10)

plant_embeddings.csv – embeddingi z nazwami (do debugowania)


**Pipeline w Plantelligence**
- Świadomie **nie łączymy wszystkiego w jeden obiekt sklearn.Pipeline**.
- Trzymamy artefakty osobno: `ohe.pkl`, `scaler.pkl`, `pca.pkl`.
- Dzięki temu każdy krok (OHE → Scaler → Binarka → PCA) jest przejrzysty i łatwy do debugowania.
- Funkcje `build_feature_matrix` i `to_embedding` odtwarzają pełny pipeline inference.
