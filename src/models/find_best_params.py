import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Définir les chemins
data_processed_path = 'data/processed'
models_path = 'models'

# Charger les données d'entraînement normalisées
X_train_scaled = pd.read_csv(os.path.join(data_processed_path, 'X_train_scaled.csv'))
y_train = pd.read_csv(os.path.join(data_processed_path, 'y_train.csv'))

# Définir le modèle et la grille de paramètres
model = RandomForestRegressor(random_state=246)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Effectuer la recherche par grille
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train.values.ravel())

# Récupérer les meilleurs paramètres
best_params = grid_search.best_params_

# Sauvegarder les meilleurs paramètres
best_params_path = os.path.join(models_path, 'best_params.pkl')
pickle.dump(best_params, best_params_path)

print("Meilleurs paramètres trouvés :", best_params)
