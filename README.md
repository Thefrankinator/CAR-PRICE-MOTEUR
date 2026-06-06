# Car Price Predictor — Moteur.ma

> Estimation du prix équitable d'une voiture d'occasion au Maroc à partir des données du marché.

**Cours :** AI Algorithms and Predictions · **Prof :** Abdellah TAHIRI, Ph.D  
**Dataset :** Moteur.ma_SDBDIA2A · **Source :** https://www.moteur.ma

---

## Problématique

Le marché de la voiture d'occasion au Maroc manque de transparence : les acheteurs risquent de surpayer, les vendeurs de sous-estimer leur véhicule. Ce projet construit un outil data-driven pour estimer un prix équitable à partir des caractéristiques du véhicule et des tendances du marché.

---

## Structure du projet

```
Car-Price-Moteur/
├── notebooks/
│   ├── A_scraping.ipynb       → Web scraping moteur.ma
│   ├── B_eda.ipynb            → Analyse exploratoire
│   ├── C_preprocessing.ipynb  → Nettoyage & preprocessing
│   ├── D_modeling.ipynb       → Entraînement des modèles
│   ├── E_evaluation.ipynb     → Évaluation & sélection
│   └── F_insights.ipynb       → Business insights
├── data/
│   ├── moteur_raw.csv         → Données brutes scrapées
│   └── moteur_clean.csv       → Données nettoyées
├── README.md
└── Practical project_second_hand_car_price_SDBDIA2A.pdf
```

---

## Features sélectionnées

Features extraites directement depuis les pages d'annonces de moteur.ma.

### Informations véhicule

| Feature | Type | Description |
|---------|------|-------------|
| `Marque` | Catégorielle | Constructeur du véhicule (ex : Dacia, Volkswagen, Toyota) |
| `Modèle` | Catégorielle | Modèle exact (ex : Clio, Golf, Yaris) |
| `Année` | Numérique | Année de mise en circulation |
| `Kilométrage` | Numérique | Kilométrage total parcouru (en km) |
| `Carburant` | Catégorielle | Type de carburant (Essence, Diesel, Hybride, Électrique) |
| `Transmission` | Catégorielle | Boîte de vitesses (Manuelle, Automatique) |
| `Carrosserie` | Catégorielle | Type de carrosserie (Berline, SUV, Break, etc.) |
| `Couleur` | Catégorielle | Couleur extérieure du véhicule |
| `Puissance fiscale` | Numérique | Puissance fiscale du moteur (en CV) |
| `Cylindrée` | Numérique | Cylindrée du moteur (en cm³) |
| `Nombre de portes` | Numérique | Nombre de portes du véhicule |
| `ville` | Catégorielle | Ville de l'annonce (ex : Casablanca, Agadir) |

### Options (binaires — 0(absent) / 1 (present))

| Feature | Description |
|---------|-------------|
| `État du véhicule` | Bon état général déclaré |
| `Airbags` | Présence d'airbags |
| `Navigation GPS` | Système de navigation intégré |
| `Ordinateur de bord` | Ordinateur de bord présent |
| `Limiteur de vitesse` | Limiteur de vitesse présent |
| `Climatisation` | Climatisation présente |
| `Intérieur cuir` | Sièges en cuir |
| `Radar de recul` | Radar de recul présent |

### Target

| Feature | Type | Description |
|---------|------|-------------|
| **`prix`** | **Numérique** | **Prix de vente en Dirham marocain (MAD)** |

---

## Pipeline

```
moteur.ma → Scraping → EDA → Preprocessing → Modèles → Évaluation → Insights
```

### A — Web Scraping
- Extraction des annonces depuis `moteur.ma/fr/voiture/achat-voiture-occasion`
- Librairies : `requests`, `BeautifulSoup`, `Selenium`
- Features collectées : Marque, Modèle, Année, Kilométrage, Carburant, Transmission, Puissance, Localisation, État, **Prix**

### B — EDA
- Distribution du prix (target), analyse des skewness
- Corrélations entre features et prix
- Insights visuels : dépréciation par âge, par kilométrage, par marque

### C — Preprocessing
- Gestion des valeurs manquantes (médiane / mode)
- Traitement des outliers (méthode IQR)
- Encodage des variables catégorielles
- Feature engineering : `âge_voiture`, `km_par_an`
- Normalisation (StandardScaler)

### D — Modèles entraînés

| Modèle | Type |
|--------|------|
| Linear Regression | Baseline |
| Ridge Regression | Régularisation L2 |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| XGBoost | Boosting optimisé |

### E — Évaluation

| Métrique | Description |
|----------|-------------|
| RMSE | Erreur quadratique moyenne (en MAD) |
| MAE | Erreur absolue moyenne |
| R² | Variance expliquée |

- Cross-validation KFold (k=5)
- Hyperparameter tuning : GridSearchCV
- Analyse des résidus + Feature importance

### F — Business Insights
- Facteurs les plus influents sur le prix
- Courbe de dépréciation (km · âge)
- Marques les mieux valorisées au Maroc
- Recommandations acheteurs / vendeurs

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost requests beautifulsoup4 selenium
```

---

## Lancer le projet

```bash
# Dans l'ordre
jupyter notebook notebooks/A_scraping.ipynb
jupyter notebook notebooks/B_eda.ipynb
jupyter notebook notebooks/C_preprocessing.ipynb
jupyter notebook notebooks/D_modeling.ipynb
jupyter notebook notebooks/E_evaluation.ipynb
jupyter notebook notebooks/F_insights.ipynb
```

---

## Résultats

*(à compléter après entraînement)*

| Modèle | RMSE | MAE | R² |
|--------|------|-----|----|
| Linear Regression | — | — | — |
| Ridge | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| **XGBoost** | — | — | — |

**Meilleur modèle :** ___ · **R² :** ___ · **RMSE :** ___ MAD
