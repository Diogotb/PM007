import numpy as np
from scipy.linalg import expm

# Definir a matriz A
A = np.array([[2, 1],
              [1, 2]])

# Ordem do Método de Padé (p, q)
p = 4
q = 4

# Calcular a exponencial de A usando o Método de Padé
def pade_expm(A, p, q):
    n = A.shape[0]
    ident = np.eye(n)
    Ap = A.dot(A)
    U = np.eye(n) + (p / q) * A
    V = ident - (1 / q) * A
    for j in range(2, p + 1):
        Ap = Ap.dot(A)
        U += (1 / np.math.factorial(j + p)) * Ap
        V += (-1 / np.math.factorial(j + q)) * Ap
    return np.linalg.solve(V, U)

# Calcular a exponencial de A usando a função expm do SciPy
exponential_actual = expm(A)

# Calcular a exponencial de A usando o Método de Padé
exponential_pade = pade_expm(A, p, q)

# Imprimir os resultados
print("Exponencial de A usando scipy.linalg.expm:\n", exponential_actual)
print("\nExponencial de A usando Método de Padé:\n", exponential_pade)
