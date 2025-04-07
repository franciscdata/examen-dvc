import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

# Définir les chemins
data_processed_path = 'data/processed'
models_path = 'models'
metrics_path = 'metrics'

# Charger les données de test normalisées et les vraies valeurs de y
X_test_scaled = pd.read_csv(os.path.join(data_processed_path, 'X_test_scaled.csv'))
y_test = pd.read_csv(os.path.join(data_processed_path, 'y_test.csv'))

# Charger le modèle entraîné avec joblib
model_path = os.path.join(models_path, 'trained_model.joblib')
model = joblib.load(model_path)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# Évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Sauvegarder les métriques
scores = {
    'mse': mse,
    'r2': r2
}
scores_path = os.path.join(metrics_path, 'scores.json')
with open(scores_path, 'w') as f:
    json.dump(scores, f)

# Créer un DataFrame avec les prédictions et les valeurs réelles
predictions_df = pd.DataFrame({'predicted_silica_concentrate': y_pred, 'actual_silica_concentrate': y_test['silica_concentrate']})
predictions_path = os.path.join(data_processed_path, 'predictions.csv')
predictions_df.to_csv(predictions_path, index=False)
