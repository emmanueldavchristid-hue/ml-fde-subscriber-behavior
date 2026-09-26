# Projet ML — Prédiction du Montant FDE et Comportement des Abonnés

Projet de **Machine Learning appliqué aux données de facturation et de consommation des abonnés**, combinant régression, classification et segmentation afin d'analyser le montant du Fonds de Développement de l'Eau (FDE), les comportements de paiement et la résiliation des abonnés.

---

## Objectifs du projet

Le projet poursuit quatre objectifs principaux :

1. **Régression** — Prédire le montant `MONT-FDE`.
2. **Classification** — Prédire le risque de `RETARD` de paiement.
3. **Classification** — Prédire le statut `RESILIE` d'un abonné.
4. **Segmentation** — Identifier différents profils d'abonnés à l'aide du clustering.
5. **Sélection de variables** — Identifier les variables les plus pertinentes pour les différents modèles.

---

## Données

Les données utilisées proviennent de plusieurs fichiers de facturation :

* `DR2.txt`
* `DR6.txt`
* `DR7.txt`
* `DR9.txt`
* `DR16.txt`
* `DR21.txt`

Les fichiers sont chargés puis fusionnés afin de constituer une base de travail unique.

Les données contiennent notamment des informations relatives :

* aux abonnés ;
* à la consommation d'eau ;
* à la facturation ;
* aux montants facturés ;
* aux dates de facturation et de règlement ;
* aux caractéristiques des abonnés ;
* aux résiliations ;
* aux zones et catégories de clientèle.

> Les fichiers de données brutes ne sont pas inclus dans le dépôt GitHub lorsqu'ils sont volumineux ou soumis à des restrictions de diffusion.

---

# Méthodologie

Le projet est organisé en plusieurs étapes :

```text
Données brutes
      │
      ▼
Exploration des données (EDA)
      │
      ▼
Nettoyage & contrôle qualité
      │
      ▼
Feature Engineering
      │
      ▼
Séparation Train / Test
      │
      ▼
Préprocessing
      │
      ├───────────────┬────────────────┐
      ▼               ▼                ▼
  Régression      Classification    Clustering
      │               │                │
      ▼               ▼                ▼
   MONT-FDE     RETARD / RESILIE    Profils clients
      │               │                │
      └───────────────┴────────────────┘
                      │
                      ▼
             Analyse des résultats
```

---

# Partie 1 — Exploration des données (EDA)

L'analyse exploratoire permet de comprendre la structure et la qualité des données avant la modélisation.

### Principales étapes

* Chargement des fichiers ;
* Fusion des différentes sources ;
* Analyse de la structure du DataFrame ;
* Analyse des types de variables ;
* Création d'un dictionnaire de données ;
* Analyse des valeurs manquantes ;
* Analyse statistique des variables quantitatives ;
* Analyse des variables qualitatives ;
* Analyse des variables cibles ;
* Visualisation des distributions ;
* Analyse des corrélations ;
* Analyse de l'évolution temporelle.

### Variables cibles

#### `MONT-FDE`

Variable quantitative utilisée pour le problème de **régression**.

Elle représente le montant du Fonds de Développement de l'Eau.

#### `RETARD`

Variable binaire utilisée pour la **classification** :

```text
0 → Paiement à temps
1 → Paiement en retard
```

Le retard est déterminé à partir du délai entre la date de facturation et la date de règlement, avec un seuil de **30 jours**. Les factures non réglées sont également considérées comme étant en retard.

#### `RESILIE`

Variable utilisée pour la **classification du statut de résiliation** :

```text
0 → Résilié
1 → Non résilié / actif
```

Lorsque nécessaire, cette variable est construite ou vérifiée à partir de `DATE-RESIL`.

#### `ENCAISSE`

Variable complémentaire :

```text
0 → Non encaissé
1 → Encaissé
```

Elle est construite à partir de la présence de `DATE-REGLT`.

---

# Partie 2 — Preprocessing et Feature Engineering

## Séparation Train / Test

Les données sont séparées selon un ratio :

```text
80 % → Train
20 % → Test
```

avec :

```python
random_state = 42
```

Lorsque cela est possible, une stratification est utilisée afin de conserver une distribution similaire des classes entre les ensembles d'entraînement et de test.

---

## Traitement des dates

Plusieurs variables temporelles sont extraites :

* année de facturation ;
* mois de facturation ;
* jour de la semaine ;
* trimestre ;
* jour du mois ;
* année d'abonnement ;
* mois d'abonnement ;
* ancienneté de l'abonné.

Une variable `ANCIENNETE` est notamment construite afin de représenter l'ancienneté de l'abonné au moment de la facturation.

---

## Variables dérivées

Plusieurs variables sont créées afin d'enrichir les données :

| Variable         | Description                                                          |
| ---------------- | -------------------------------------------------------------------- |
| `ABONNE`         | Identifiant construit à partir de plusieurs informations de l'abonné |
| `RATIO_SOCIAL`   | Proportion de consommation sociale                                   |
| `MONTANT_M3`     | Montant TTC rapporté au volume facturé                               |
| `CONSO_SUP_FACT` | Indicateur lorsque la consommation dépasse le volume facturé         |
| `POURCENT_FDE`   | Part du FDE dans le montant TTC                                      |
| `DELAI_PAIEMENT` | Nombre de jours entre facturation et règlement                       |
| `ENCAISSE`       | Indicateur d'encaissement                                            |
| `RETARD`         | Indicateur de retard de paiement                                     |

Certaines variables dérivées sont ensuite exclues des modèles lorsqu'elles introduisent une **fuite de données (data leakage)**.

---

## Traitement des valeurs manquantes

La stratégie appliquée est la suivante :

* suppression des variables présentant plus de 50 % de valeurs manquantes ;
* imputation des variables numériques par la **médiane calculée sur le train** ;
* imputation des variables catégorielles par le **mode calculé sur le train**.

Les paramètres calculés sur l'ensemble d'entraînement sont ensuite appliqués à l'ensemble de test.

---

## Encodage des variables catégorielles

Selon leur cardinalité :

* variables binaires → Label Encoding ;
* variables comportant peu de modalités → One-Hot Encoding ;
* variables à forte cardinalité → Label Encoding.

Les variables identifiantes ou présentant un risque de fuite d'information sont exclues des features lorsque nécessaire.

---

# Partie 3 — Modélisation

## 1. Régression — Prédiction de `MONT-FDE`

L'objectif est de prédire le montant du FDE à partir des informations disponibles sur l'abonné, sa consommation et son contexte de facturation.

Une attention particulière est portée au **data leakage**.

Les variables directement liées au montant cible, telles que certains montants calculés à partir du FDE, sont exclues du modèle.

Une approche intermédiaire est utilisée avec `CUBFAC`, représentant le volume total facturé, tout en excluant les cubages détaillés afin d'éviter de reconstruire directement le montant du FDE.

### Modèles testés

Six modèles de régression sont comparés :

1. **Régression linéaire — MCO**
2. **Ridge**
3. **Lasso**
4. **Elastic Net**
5. **PCR — Principal Component Regression**
6. **PLS — Partial Least Squares**

### Métriques utilisées

Les modèles sont évalués avec :

* **R²**
* **RMSE**
* **MAE**
* validation croisée **5-fold**

Le tableau de comparaison est généré directement dans le notebook.

---

# 2. Classification — Prédiction du retard

L'objectif est de prédire si une facture présente un risque de retard de paiement.

La variable cible est :

```text
RETARD
0 → À temps
1 → En retard
```

Une attention particulière est portée à la prévention du **data leakage**.

Les variables révélant directement qu'un paiement a déjà été effectué sont notamment exclues :

* `DATE-REGLT`
* `AAENC`
* `MMENC`
* `MMENC_CLEAN`
* `DELAI_PAIEMENT`
* `ENCAISSE`

L'objectif est ainsi de construire une prédiction basée sur les informations disponibles **avant le règlement**.

Les performances sont évaluées à l'aide de métriques adaptées à la classification.

---

# 3. Classification — Prédiction de la résiliation

Le deuxième problème de classification consiste à identifier le statut de résiliation d'un abonné.

La variable cible est :

```text
RESILIE
0 → Résilié
1 → Actif / non résilié
```

La variable est contrôlée à partir de `DATE-RESIL` afin de vérifier la cohérence des données.

---

# 4. Segmentation des abonnés

Une étape de **clustering K-Means** est également réalisée afin d'identifier différents profils d'abonnés.

Le processus comprend :

1. Sélection des variables pertinentes ;
2. Normalisation des variables ;
3. Analyse de plusieurs valeurs de `K` ;
4. Utilisation de l'inertie intra-classe ;
5. Analyse du coefficient de silhouette ;
6. Application du clustering final ;
7. Analyse des centres de gravité des clusters.

Le modèle final utilise **4 clusters** pour construire différents profils types d'abonnés.

Les profils sont ensuite analysés à travers leurs caractéristiques moyennes et leur taux de résiliation.

---

# Prévention du Data Leakage

Une attention particulière est portée à la séparation entre les informations disponibles au moment de la prédiction et celles qui ne sont connues qu'après l'événement à prédire.

Par exemple, pour la prédiction du retard, les informations directement liées au règlement sont exclues.

Pour la prédiction du `MONT-FDE`, les variables permettant de reconstruire directement le montant cible sont également retirées lorsque nécessaire.

Cette étape permet d'obtenir une évaluation plus représentative du comportement des modèles dans un contexte de prédiction.

---

# Technologies utilisées

### Langage

* Python

### Analyse de données

* Pandas
* NumPy

### Visualisation

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Modèles

* Linear Regression
* Ridge
* Lasso
* Elastic Net
* PCR
* PLS
* K-Means

### Environnement

* Jupyter Notebook

---

# Structure du projet

```text
ml-fde-subscriber-behavior/
│
├── Projet_ML_FDE_Complete v2.ipynb
│
├── .gitignore
│
├── Data/
│   ├── DR2.txt
│   ├── DR6.txt
│   ├── DR7.txt
│   ├── DR9.txt
│   ├── DR16.txt
│   └── DR21.txt
│
└── README.md
```

> Le dossier `Data/` peut être exclu du dépôt lorsque les fichiers sont trop volumineux ou ne peuvent pas être publiés.

---

# Équipe

Projet réalisé par :

1. **MOUHI CHRIST-EMMANUEL**
2. **FOFANA ADELPHE VIANNEY**
3. **MEITE YOUSSOUF**
4. **COULIBALY SEGNINDENIN OUMAR**

---

# Résultats

Le notebook contient les résultats détaillés des différentes étapes :

* statistiques descriptives ;
* analyse des valeurs manquantes ;
* distributions des variables ;
* corrélations ;
* performances des modèles de régression ;
* performances des modèles de classification ;
* analyse des variables importantes ;
* profils issus du clustering.

Les résultats sont générés directement lors de l'exécution du notebook afin de conserver une analyse reproductible.

---

# Perspectives

Plusieurs améliorations peuvent être envisagées :

* optimisation des hyperparamètres ;
* comparaison avec des modèles non linéaires tels que Random Forest, XGBoost ou LightGBM ;
* amélioration du traitement du déséquilibre des classes ;
* sélection automatique des variables ;
* analyse plus poussée des erreurs de prédiction ;
* interprétabilité des modèles avec SHAP ;
* mise en place d'une API de prédiction ;
* création d'une interface de visualisation des résultats ;
* industrialisation du pipeline de Machine Learning.

---

## Auteurs

**MOUHI CHRIST-EMMANUEL**
**FOFANA ADELPHE VIANNEY**
**MEITE YOUSSOUF**
**COULIBALY SEGNINDENIN OUMAR**

Projet Machine Learning — Prédiction du FDE et comportement des abonnés.
