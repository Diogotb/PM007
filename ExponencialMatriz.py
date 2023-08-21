import numpy as np
from scipy.linalg import expm

# Definir a matriz A
A = np.array([[2, 1, 3],
              [3, 1, 2],
              [1, 2, 3]])

# Definir o número de termos na série de Taylor
num_terms = 10

# Calcular a exponencial de A usando a série de Taylor
exponential_approx = np.eye(A.shape[0])
current_term = np.eye(A.shape[0])
for i in range(1, num_terms + 1):
    current_term = np.dot(current_term, A) / i
    exponential_approx += current_term

# Calcular a exponencial de A usando a função expm do SciPy
exponential_actual = expm(A)

# Imprimir os resultados
print("Exponencial de A usando série de Taylor:\n", exponential_approx)
print("\nExponencial de A usando scipy.linalg.expm:\n", exponential_actual)
