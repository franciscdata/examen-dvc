import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Définir les chemins
data_raw_path = 'data/raw'
data_processed_path = 'data/processed'

# Charger le dataset
df = pd.read_csv(os.path.join(data_raw_path, 'raw.csv'))

# Définir la variable cible et les features
X = df.drop('silica_concentrate', axis=1)
y = df['silica_concentrate']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=246)

# Sauvegarder les datasets
X_train.to_csv(os.path.join(data_processed_path, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(data_processed_path, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(data_processed_path, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(data_processed_path, 'y_test.csv'), index=False)

