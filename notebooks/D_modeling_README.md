# Partie D — Modélisation : estimation du prix des voitures d'occasion

Documentation du notebook `notebooks/D_modeling.ipynb`.
**Périmètre :** modélisation (mon rôle dans le projet d'équipe). Le notebook construit,
compare et optimise plusieurs modèles de régression pour prédire le prix d'une voiture
d'occasion (en MAD), puis évalue le meilleur sur un jeu de test mis de côté.

**Hypothèse d'entrée :** les données ont passé une validation de cohérence (valeurs de
prix et de kilométrage physiquement impossibles écartées). La logique de nettoyage
complète appartient à l'étape *preprocessing* ; ce notebook applique une revérification
défensive mais n'en est pas propriétaire.

---

## Démarche en bref

```
Audit → Split (stratifié, cible log) → Pipeline anti-fuite → Baseline 5 modèles
      → Sélection de features → Tuning → Évaluation test → Sauvegarde
```

Trois principes guident tout le notebook :

1. **Pas de fuite de données.** Toute transformation qui *apprend* des paramètres
   (imputation, scaling, encodage par cible) vit *dans* le pipeline et est ré-apprise
   sur chaque fold de validation croisée — jamais sur les données de test.
2. **Cible en log.** Les prix sont fortement asymétriques ; on modélise `log1p(prix)`
   et on re-transforme les métriques en dirhams (`expm1`) pour l'interprétation.
3. **Sélection sur la validation croisée, test touché une seule fois.** Le test ne sert
   qu'à confirmer la généralisation du modèle déjà choisi.

---

## 1. Audit et nettoyage

L'audit des données brutes a révélé plusieurs problèmes : `Kilométrage` au format texte,
prix manquants, colonnes très incomplètes (`Cylindrée` ~100 %, `Carrosserie` ~59 %,
`Couleur` ~44 %), années aberrantes, et forte cardinalité de `Marque`/`Modèle`/`ville`.

La fonction `load_clean_data()` applique un nettoyage **déterministe** (sans paramètre
appris, donc sans fuite) : parsing du kilométrage, mise au format des valeurs manquantes,
filtrage des valeurs hors-bornes, création de `age_voiture` et `km_par_an`, et abandon
des colonnes inutilisables.

### Le point décisif : les valeurs aberrantes

Un premier passage de modélisation sur données **non filtrées** plafonnait tous les
modèles à R² ≈ 0,38, les modèles d'ensemble ne faisant pas mieux qu'une régression
linéaire — signal classique d'un bruit qui noie l'information. Le diagnostic a révélé
des valeurs physiquement impossibles (kilométrage jusqu'à ~2,1 milliards de km, prix
jusqu'à 58 M MAD). Une fois ces bornes appliquées, l'effet est sans ambiguïté :

| Données | R² (modèle rapide, illustratif) |
|---------|--------------------------------|
| Sans filtres de bornes | 0,35 |
| Avec filtres de bornes | 0,77 |

*(Chiffres illustratifs : Random Forest, 3-fold. Voir le baseline complet ci-dessous.)*

---

## 2. Séparation train / test

Split 80/20, **stratifié sur 5 tranches de prix** (`pd.qcut`) pour que train et test
aient des distributions de prix comparables (on stratifie sur des tranches, jamais sur
la cible continue). Les médianes de prix train/test sont quasi identiques, confirmant
un découpage équilibré.

---

## 3. Pipeline de preprocessing anti-fuite

Cœur méthodologique. Un `ColumnTransformer` traite chaque type de colonne et reste
encapsulé dans le pipeline, donc ré-appris par fold :

| Colonnes | Traitement | Raison |
|----------|-----------|--------|
| Numériques | médiane → StandardScaler | scaling utile au linéaire, inoffensif pour les arbres |
| `Marque`, `Modèle`, `ville` | TargetEncoder | forte cardinalité : le one-hot exploserait la dimension |
| `Carburant`, `Transmission` | mode → OneHot | peu de modalités |
| Options binaires (0/1) | passthrough | déjà au bon format |

---

## 4. Baseline — 5 modèles en validation croisée 5-fold

| Modèle | R² (CV) | RMSE (log) |
|--------|---------|-----------|
| RandomForest | 0,776 | 0,307 |
| GradientBoosting | 0,776 | 0,307 |
| XGBoost | 0,766 | 0,314 |
| Ridge | 0,735 | 0,334 |
| LinearRegression | 0,735 | 0,334 |

Les modèles d'arbres dépassent les linéaires (~0,78 vs ~0,74), cohérent avec une
relation non-linéaire entre prix, âge, kilométrage et marque. Faible écart-type entre
folds (~±0,02–0,03) : résultats stables. Les trois arbres se tiennent en ~0,01 — trop
proches pour les départager sans optimisation.

---

## 5. Sélection de features (RFE)

Test de RFE sur le modèle linéaire uniquement (les arbres font une sélection implicite ;
les envelopper coûterait des centaines de ré-entraînements pour un gain nul).

**Résultat : Ridge + RFE (10 features) = R² 0,727, contre 0,735 pour Ridge complet.**
La sélection *dégrade* légèrement la performance. Résultat informatif, pas un échec :
toutes les features portent de l'information, aucune n'est redondante au point de nuire.
On conserve donc l'ensemble complet des features — choix à la fois plus simple et plus
performant.

---

## 6. Optimisation des hyperparamètres

`RandomizedSearchCV`, 25 combinaisons par modèle d'arbre, preprocessing maintenu dans
le pipeline (anti-fuite jusque dans le tuning).

| Modèle | RMSE (log) baseline | RMSE (log) tuned | R² (CV) tuned |
|--------|---------------------|------------------|---------------|
| **XGBoost** | 0,314 | **0,295** | **0,793** |
| RandomForest | 0,307 | 0,298 | 0,789 |
| GradientBoosting | 0,307 | 0,301 | 0,784 |

**Observation clé : XGBoost, troisième au baseline, devient le meilleur après tuning** —
exactement pourquoi on optimise avant de choisir. Modèle retenu : **XGBoost**, avec des
paramètres bien régularisés (profondeur 5, learning rate 0,05, sous-échantillonnage
lignes et colonnes à 0,8), ce qui annonce une bonne généralisation.

---

## 7. Évaluation finale sur le test

Le test, mis de côté au split, n'est touché qu'ici, une seule fois, avec le modèle déjà
choisi.

| Métrique | Valeur |
|----------|--------|
| R² (échelle log) | 0,78 |
| R² (MAD) | 0,71 |
| MAE | ~43 800 MAD |
| RMSE | ~106 300 MAD |

- **R² test (0,78) ≈ R² validation croisée (0,79)** → pas de surapprentissage, le modèle
  généralise.
- **RMSE ≫ MAE** → une minorité de voitures (haut de gamme) concentre les grosses
  erreurs ; `expm1` amplifie l'erreur sur les grands prix.

### Limite du modèle (honnête)

Une MAE de ~44 000 MAD représente ~20 % d'erreur typique sur un prix médian de ~200 000
MAD. Le modèle est donc adapté à **détecter une sur- ou sous-évaluation** (un prix
nettement hors bande), pas à fournir une estimation exacte au dirham près. Cette limite
vient surtout de la **richesse des features** (pas de finition, d'état détaillé, d'options
complètes) plutôt que du choix du modèle — linéaires et arbres plafonnent ensemble, signe
d'un plafond dans les données. Le modèle est plus fiable sur le marché grand public que
sur les véhicules rares et chers, faute de signal d'apprentissage suffisant dans cette
tranche.

---

## 8. Sauvegarde et reproduction

Le **pipeline complet** (preprocessing ajusté + modèle) est sérialisé via `joblib` dans
`models/`. Sauvegarder le pipeline entier — et non le seul XGBoost — garantit que
`joblib.load(...).predict(X)` fonctionne de bout en bout sur des données au format de
`load_clean_data()`.

Conformément au `.gitignore` du projet, `models/` n'est pas suivi par git (artefact
régénérable). Pour reproduire le modèle : installer `requirements.txt`, puis
**Restart & Run All**. Le tuning (section 6) prend quelques minutes.

---

## Reproductibilité

- `random_state=42` partout (split, CV, modèles, recherche).
- Versions clés : scikit-learn 1.5.0, xgboost 2.0.3 (voir `requirements.txt`).
- Données : `data/raw_moteur.csv` (suivi par git).