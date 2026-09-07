# Examen DVC

Pipeline DVC de prediction de la concentration en silice.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execution

```bash
dvc repro
```

## Etapes

- split : separation des donnees en X_train, X_test, y_train, y_test
- normalize : normalisation des donnees avec StandardScaler
- gridsearch : recherche des meilleurs parametres du GradientBoostingRegressor
- training : entrainement du modele
- evaluate : evaluation du modele et calcul des metriques
