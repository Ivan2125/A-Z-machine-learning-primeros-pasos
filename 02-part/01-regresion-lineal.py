# Regresión Lineal

# Importar las librearías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Ahora se divide (Train/Test) el dataset para la realización del modelo
from sklearn.model_selection import train_test_split

# Se toma 1/3 para la divisón indicando que por 3 sueldos me separe uno para el test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=0
)

# Entrenamos el modelo de Regresión Lineal
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Teniendo el modelo de Regresión entrenado, pasamos a predecir los datos del dataset del Test
y_pred = regressor.predict(X_test)

# Ahora vamos a observar los resultados del entrenamiento del modelo de Regresión
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Suedo vs Años de experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.show()

# Ahora vamos a observar los resultados del test del modelo de Regresión
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Suedo vs Años de experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.show()

# Realizamos las predicciones
y_test_pred = regressor.predict(X_test)
y_train_pred = regressor.predict(X_train)

# Evaluamos el training y el test del modelo a través de r cuadrado
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_test_pred)  # test
r2training = r2_score(y_train, y_train_pred)  # train
print("El coeficiente de determinación del regresor es:", r2)
print("El coeficiente de determinación del regresor training es:", r2training)
