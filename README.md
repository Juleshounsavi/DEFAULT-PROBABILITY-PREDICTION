# Prédiction de la Probabilité de Défaut de Prêt

Application web de credit scoring qui estime la probabilité qu'un emprunteur soit en défaut de paiement (retard ≥ 90 jours) dans les deux prochaines années, à partir de ses caractéristiques financières et personnelles. Le modèle est entraîné sur le dataset Kaggle **[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)** et exposé via une interface Flask.

## Contexte

Ce projet reprend un ancien concours Kaggle : construire un modèle de scoring capable de prédire, à partir de 10 variables financières et démographiques, si un individu connaîtra un défaut de paiement grave dans les deux années suivantes. Le dataset comprend 150 000 lignes et est fortement déséquilibré (environ 93% de non-défauts / 7% de défauts).

## Dataset

10 variables prédictives, dont :

| Variable | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Taux d'utilisation des lignes de crédit renouvelables |
| `age` | Âge de l'emprunteur |
| `NumberOfTime30-59DaysPastDueNotWorse` | Nombre de retards de 30-59 jours |
| `DebtRatio` | Ratio d'endettement |
| `MonthlyIncome` | Revenu mensuel |
| `NumberOfOpenCreditLinesAndLoans` | Nombre de lignes de crédit et prêts ouverts |
| `NumberOfTimes90DaysLate` | Nombre de retards de 90 jours ou plus |
| `NumberRealEstateLoansOrLines` | Nombre de prêts/lignes immobiliers |
| `NumberOfTime60-89DaysPastDueNotWorse` | Nombre de retards de 60-89 jours |
| `NumberOfDependents` | Nombre de personnes à charge |

Variable cible : `SeriousDlqin2yrs` (0 = pas de défaut, 1 = défaut).

## Méthodologie

1. **Nettoyage** : imputation de `MonthlyIncome` (19.8% de valeurs manquantes, remplacées par la médiane) et `NumberOfDependents` (2.6%, remplacées par le mode) ; traitement des valeurs aberrantes colonne par colonne (seuils métier + méthode IQR), remplacées par la médiane plutôt que supprimées pour ne pas perdre de lignes.
2. **Analyse exploratoire** : distribution des variables, matrice de corrélation, corrélation individuelle de chaque feature avec la variable cible.
3. **Modélisation** : comparaison de trois modèles sur un split 70/30: régression logistique, Random Forest (`class_weight="balanced"`), XGBoost.
4. **Sélection** : XGBoost retenu sur la base du ROC-AUC, puis exporté (`joblib`) et servi par une application Flask avec formulaire HTML.

## Stack technique

| Composant | Usage |
|---|---|
| Python (pandas, numpy) | Manipulation des données |
| scikit-learn | Régression logistique, Random Forest, métriques |
| XGBoost | Modèle retenu en production |
| matplotlib, seaborn | Visualisation exploratoire |
| Flask | Application web / API de prédiction |
| joblib | Sérialisation du modèle |

## Résultats


| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Régression logistique | 0.9355 | 0.5622 | 0.1534 | 0.2410 | 0.8076 |
| Random Forest (balanced) | 0.9323 | 0.4795 | 0.1517 | 0.2305 | 0.8287 |
| **XGBoost (retenu)** | **0.9364** | **0.5766** | **0.1790** | **0.2732** | **0.8596** |
 

Le ROC-AUC de 0.86 indique une bonne capacité de séparation globale des classes. En revanche, le recall reste faible sur les trois modèles.
## Pistes d'amélioration

Je documente ces pistes explicitement, parce qu'elles sont le point le plus intéressant du projet d'un point de vue méthodologique :

- **Recall à améliorer (17.9% pour XGBoost)** : le modèle ne détecte qu'environ 1 défaut sur 5. Dans un contexte réel de credit scoring, c'est le type d'erreur le plus coûteux (faux négatif = crédit accordé à un client qui ne remboursera pas). Le seuil de décision par défaut (0.5) n'a pas encore été ajusté, et le déséquilibre des classes n'a pas été traité pour XGBoost ni pour la régression logistique (seul Random Forest utilise `class_weight="balanced"`).
- **Mise à l'échelle des features** à ajouter pour la régression logistique, dont les variables sont sur des échelles très différentes (`age` vs `MonthlyIncome` vs `DebtRatio`) — ceci pénalise probablement ce modèle en particulier.
- **Validation croisée et recherche d'hyperparamètres** à mettre en place : les modèles sont pour l'instant évalués sur un unique split train/test, et les hyperparamètres XGBoost ne sont pas issus d'une recherche systématique.

## Architecture du projet

```
DEFAULT-PROBABILITY-PREDICTION/
├── PROJET_DE_FIN_DE_MODULE_PYTHON.ipynb   → EDA, nettoyage, modélisation, évaluation
├── cs-training.csv                        → Dataset Kaggle "Give Me Some Credit"
├── deploiement/
│   ├── app.py                             → Application Flask (route /, /predict)
│   ├── model.pkl                          → Modèle XGBoost sérialisé (joblib)
│   ├── templates/index.html               → Formulaire de saisie
│   └── static/style.css                   → Styles
└── DEFAULT PROBABILITY PROJECT.pdf        → Énoncé du projet
```

## Installation et lancement

```bash
git clone https://github.com/Juleshounsavi/DEFAULT-PROBABILITY-PREDICTION.git
cd DEFAULT-PROBABILITY-PREDICTION

# Créer et activer un environnement virtuel
python -m venv venv
# Windows (PowerShell) : .\venv\Scripts\Activate.ps1
# Windows (Git Bash)    : source venv/Scripts/activate
# macOS / Linux         : source venv/bin/activate

# Installer les dépendances
pip install flask numpy scikit-learn xgboost joblib

# Lancer l'application
cd deploiement
python app.py
```

L'application est ensuite accessible sur `http://127.0.0.1:5000/`.

## Contributeurs
- **HOUNSAVI Jules Koffi**
- **TARDANE Hafsa**  
