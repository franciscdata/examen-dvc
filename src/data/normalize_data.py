import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Définir les chemins
data_processed_path = 'data/processed'

# Charger les datasets d'entraînement et de test
X_train = pd.read_csv(os.path.join(data_processed_path, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(data_processed_path, 'X_test.csv'))

# Initialiser le scaler
scaler = StandardScaler()

# Normaliser les données d'entraînement
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Normaliser les données de test en utilisant le scaler entraîné sur les données d'entraînement
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Sauvegarder les datasets normalisés
X_train_scaled_df.to_csv(os.path.join(data_processed_path, 'X_train_scaled.csv'), index=False)
X_test_scaled_df.to_csv(os.path.join(data_processed_path, 'X_test_scaled.csv'), index=False)
