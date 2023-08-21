import numpy as np
from scipy.linalg import expm
from scipy.linalg import norm

# Definir a matriz A
A = np.array([[2, 1],
              [1, 2]])

# Número de Krylov subspace
krylov_size = 10

# Vetor inicial
v0 = np.random.rand(A.shape[0])

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

# Função que realiza o método de Arnoldi
def arnoldi(A, v, m):
    V = np.zeros((len(v), m + 1))
    H = np.zeros((m + 1, m))
    V[:, 0] = v / np.linalg.norm(v)
    for j in range(m):
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w -= H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] == 0:
            break
        V[:, j + 1] = w / H[j + 1, j]
    return V[:, :j + 1], H[:j + 1, :j]

# Realizar o método de Arnoldi
V, H = arnoldi(A, v0, krylov_size)

# Calcular a exponencial de A usando scipy.linalg.expm
exponential_actual = expm(A)

# Calcular a exponencial de A aproximada usando a expansão de Pade
exponential_approx = pade_expm(A, p, q)

# Imprimir os resultados
print("Exponencial de A usando scipy.linalg.expm:\n", exponential_actual)
print("\nExponencial de A aproximada usando Método de Arnoldi:\n", exponential_approx)
