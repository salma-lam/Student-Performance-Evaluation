                        #### La performance des etudiants ####
                               ## SVM (Multi-classe) ##

# # Collecte des données :
import pandas as pd

# Charger les données depuis le fichier CSV
data = pd.read_csv("StudentsPerformance.csv")



# # Prétraitement des données :
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Supprimer les lignes avec des valeurs manquantes
data.dropna(inplace=True)

# Convertir les variables catégoriques en variables numériques
label_encoders = {}
categorical_columns = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Normaliser les caractéristiques numériques
numeric_columns = ["reading score", "writing score"]

scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])



# # Division des données :
from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test
X = data.drop(columns=["math score"])
y = data["math score"]  

# Utiliser une répartition de 80% pour l'entraînement et 20% pour les tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Afficher la taille des ensembles d'entraînement et de test
print("Taille de l'ensemble d'entraînement:", len(X_train))
print("Taille de l'ensemble de test:", len(X_test))



# # Entraînement des modèles :
from sklearn.svm import SVR

# Créer un modèle SVR pour la régression
svm_regressor = SVR(kernel='linear')  

# Entraîner le modèle sur l'ensemble d'entraînement
svm_regressor.fit(X_train, y_train)



# # Évaluation des modèles :
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

# Prédire sur l'ensemble de test
y_pred = svm_regressor.predict(X_test)

# Calculer les métriques pour évaluer le modèle de régression
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)  
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Afficher les métriques
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)




# # Vitesse d'exécution :
import time

# Chronométrer le temps d'entraînement
start_time = time.time()
svm_regressor.fit(X_train, y_train)
training_time = time.time() - start_time

# Chronométrer le temps de prédiction
start_time = time.time()
y_pred = svm_regressor.predict(X_test)
prediction_time = time.time() - start_time

# Afficher la vitesse d'exécution
print("Temps d'entraînement du modèle SVM:", training_time, "secondes")
print("Temps de prédiction du modèle SVM:", prediction_time, "secondes")




# # La courbe des valeurs prédites par rapport aux vraies valeurs
import matplotlib.pyplot as plt

# Tracer la courbe des valeurs prédites par rapport aux vraies valeurs
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True Values vs Predicted Values')
plt.grid(True)
plt.show()



# # Graphe nombre de valeurs pour chaque math score :
import seaborn as sns
sns.catplot(x='math score', data=data, kind='count')
# Ajouter un titre et des étiquettes d'axe
plt.title('Nombre de valeurs pour chaque math score')
plt.xlabel('Math Score')
plt.ylabel('Nombre de valeurs')
# Afficher le graphique
plt.show()

