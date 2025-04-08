import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Définir les chemins
data_processed_path = 'data/processed'
models_path = 'models'

# Charger les données d'entraînement normalisées
X_train_scaled = pd.read_csv(os.path.join(data_processed_path, 'X_train_scaled.csv'))
y_train = pd.read_csv(os.path.join(data_processed_path, 'y_train.csv'))

# Charger les meilleurs paramètres
best_params_path = os.path.join(models_path, 'best_params.pkl')
best_params = pickle.load(best_params_path)

# Initialiser et entraîner le modèle avec les meilleurs paramètres
model = RandomForestRegressor(**best_params, random_state=246)
model.fit(X_train_scaled, y_train.values.ravel())

# Sauvegarder le modèle entraîné
model_path = os.path.join(models_path, 'trained_model.pkl')
pickle.dump(model, model_path)
