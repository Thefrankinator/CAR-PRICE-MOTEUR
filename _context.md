# Car-Price-Moteur — Context

## Description

Mini-projet d'exam pratique — **Cours : AI Algorithms and Predictions**  
**Prof :** Abdellah TAHIRI, Ph.D  
**Dataset :** Moteur.ma_SDBDIA2A  
**Source des données :** https://www.moteur.ma/fr/voiture/achat-voiture-occasion

**Objectif :** Construire un outil data-driven pour estimer le prix équitable d'une voiture d'occasion au Maroc, à partir de features historiques et des tendances du marché.

---

## Cahier des charges (6 parties)

| Partie | Contenu |
|--------|---------|
| **A** | Web Scraping — moteur.ma (requests / BeautifulSoup / Selenium) |
| **B** | EDA — comprendre, visualiser, extraire des insights |
| **C** | Preprocessing — missing values, outliers, encoding, feature engineering, normalisation |
| **D** | Modèles — définir features/target, train-test split, fit, prédictions |
| **E** | Évaluation — comparaison modèles, hyperparameter tuning, cross-validation, sélection |
| **F** | Business insights — recommandations pour acheteurs/vendeurs |

---

## Features à collecter

| Feature | Type |
|---------|------|
| Marque | Catégorielle |
| Modèle | Catégorielle |
| Année | Numérique |
| Kilométrage | Numérique |
| Carburant | Catégorielle |
| Transmission | Catégorielle |
| Puissance moteur | Numérique |
| Localisation | Catégorielle |
| État | Catégorielle |
| **Prix** | **Target (numérique)** |

---

## Structure du projet

```
Car-Price-Moteur/
├── _context.md                         ← ce fichier
├── Practical project_second_hand_car_price_SDBDIA2A.pdf   ← énoncé (à déplacer depuis Inbox)
├── data/
│   └── moteur_raw.csv                  ← données scrapées
├── notebooks/
│   ├── A_scraping.ipynb
│   ├── B_eda.ipynb
│   ├── C_preprocessing.ipynb
│   ├── D_modeling.ipynb
│   ├── E_evaluation.ipynb
│   └── F_insights.ipynb
└── requirements.txt
```

---

## Stack technique

- **Scraping :** `requests`, `BeautifulSoup`, `Selenium` (si JS nécessaire)
- **Data :** `pandas`, `numpy`
- **Visualisation :** `matplotlib`, `seaborn`, `plotly`
- **ML :** `scikit-learn` (LinearRegression, Ridge, RandomForest, GradientBoosting, XGBoost)
- **Évaluation :** RMSE, MAE, R² · GridSearchCV · KFold cross-validation

---

## Status

- [x] PDF déplacé depuis Inbox
- [ ] Scraping moteur.ma → `data/moteur_raw.csv`
- [ ] A_scraping.ipynb complété
- [ ] B_eda.ipynb complété
- [ ] C_preprocessing.ipynb complété
- [ ] D_modeling.ipynb complété
- [ ] E_evaluation.ipynb complété (meilleur modèle sélectionné)
- [ ] F_insights.ipynb — business insights rédigés
- [ ] Rendu au prof

---

## Décisions techniques

- **Scraping :** Commencer par `requests + BeautifulSoup`. Passer à `Selenium` si le site charge en JS.
- **Target :** Prix en MAD (Dirham marocain) — vérifier l'unité sur moteur.ma
- **Outliers :** IQR + log-transform du prix si distribution très skewed
- **Modèles à comparer :** LinearRegression · Ridge · RandomForest · GradientBoostingRegressor · XGBoost
- **Métrique principale :** RMSE (en MAD) + R²

---

**Cours :** AI Algorithms and Predictions — SDBDIA 2A  
**PDF source :** `Inbox/Practical project_second_hand_car_price_SDBDIA2A.pdf` → à déplacer ici
