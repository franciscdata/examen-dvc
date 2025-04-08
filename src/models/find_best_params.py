import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import yaml

# Définir les chemins
data_processed_path = 'data/processed'
models_path = 'models'
params_path = 'params.yaml'

# Charger les paramètres depuis params.yaml
with open(params_path, 'r') as f:
    params = yaml.safe_load(f)
grid_search_params = params['grid_search']

# Charger les données d'entraînement normalisées
X_train_scaled = pd.read_csv(os.path.join(data_processed_path, 'X_train_scaled.csv'))
y_train = pd.read_csv(os.path.join(data_processed_path, 'y_train.csv'))

# Définir le modèle
model_name = grid_search_params['model']
if model_name == 'RandomForestRegressor':
    model = RandomForestRegressor(random_state=grid_search_params['random_state'])
else:
    raise ValueError(f"Modèle non supporté : {model_name}")

# Définir la grille de paramètres depuis params.yaml
param_grid = grid_search_params['param_grid']

# Effectuer la recherche par grille
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=grid_search_params['cv'],
    scoring=grid_search_params['scoring'],
    n_jobs=grid_search_params['n_jobs']
    )
grid_search.fit(X_train_scaled, y_train.values.ravel())

# Récupérer les meilleurs paramètres
best_params = grid_search.best_params_

# Sauvegarder les meilleurs paramètres
best_params_path = os.path.join(models_path, 'best_params.pkl')
with open(best_params_path, 'wb') as f:
    pickle.dump(best_params, f)

print("Meilleurs paramètres trouvés :", best_params)
