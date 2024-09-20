                        #### La performance des etudiants ####
                         ## Arbre de decision (Regression) ##

# # Collecte des données :
import pandas as pd

# Charger les données depuis le fichier CSV
data = pd.read_csv("StudentsPerformance.csv")



# # Prétraitement des données :
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Supprimer les valeurs manquantes
data.dropna(inplace=True)

# Convertir les variables catégoriques en variables numériques
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Normaliser les caractéristiques si nécessaire
# Par exemple, on va normaliser les scores de lecture et écriture
columns_to_normalize = ['reading score', 'writing score']
scaler = StandardScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])



# # Division des données :
from sklearn.model_selection import train_test_split

# Diviser les données en caractéristiques (X) et la colonne cible (y)
X = data.drop("math score", axis=1)  # Caractéristiques
y = data["math score"]  # Colonne cible

# Diviser l'ensemble de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# test_size=0.2 signifie que 20% des données seront utilisées pour le test
# random_state=42 permet de garantir que la division soit toujours la même pour la reproductibilité des résultats

# Afficher la taille des ensembles d'entraînement et de test
print("Taille de l'ensemble d'entraînement:", len(X_train))
print("Taille de l'ensemble de test:", len(X_test))





# # Entraînement des modèles :
from sklearn.tree import DecisionTreeRegressor

# Créer un modèle d'arbres de décision pour la régression
decision_tree_model = DecisionTreeRegressor(random_state=42)

# Entraîner le modèle d'arbres de décision sur l'ensemble d'entraînement
decision_tree_model.fit(X_train, y_train)

print("Le modèle d'arbres de décision pour la régression a été entraîné avec succès.")



# # Évaluation des modèles : 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Prédire les valeurs sur l'ensemble de test pour le modèle d'arbre de décision
decision_tree_predictions = decision_tree_model.predict(X_test)

# Calculer le RMSE pour le modèle d'arbre de décision
decision_tree_rmse = np.sqrt(mean_squared_error(y_test, decision_tree_predictions))

# Calculer le MSE
decision_tree_mse = mean_squared_error(y_test, decision_tree_predictions)

# Calculer le MAE
decision_tree_mae = mean_absolute_error(y_test, decision_tree_predictions)

# Calculer le R²
decision_tree_r2 = r2_score(y_test, decision_tree_predictions)

print("RMSE pour le modèle d'arbre de décision:", decision_tree_rmse)
print("MSE pour le modèle d'arbre de décision:", decision_tree_mse)
print("MAE pour le modèle d'arbre de décision:", decision_tree_mae)
print("R² pour le modèle d'arbre de décision:", decision_tree_r2)



# # Vitesse d'exécution :
import time

# Chronométrer le temps d'entraînement
start_time = time.time()
decision_tree_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Chronométrer le temps de prédiction
start_time = time.time()
decision_tree_predictions = decision_tree_model.predict(X_test)
prediction_time = time.time() - start_time

# Afficher la vitesse d'exécution
print("Temps d'entraînement du modèle:", training_time, "secondes")
print("Temps de prédiction du modèle:", prediction_time, "secondes")



# # La courbe des valeurs prédites par rapport aux vraies valeurs
import matplotlib.pyplot as plt

# Prédire les valeurs sur l'ensemble de test pour le modèle d'arbre de décision
decision_tree_predictions = decision_tree_model.predict(X_test)

# Tracer la courbe des valeurs prédites par rapport aux vraies valeurs
plt.figure(figsize=(10, 6))
plt.scatter(y_test, decision_tree_predictions, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
plt.title("Valeurs Prédites vs. Vraies Valeurs (Arbre de Décision pour la Régression)")
plt.xlabel("Vraies Valeurs")
plt.ylabel("Valeurs Prédites")
plt.show()

