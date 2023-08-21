import numpy as np
import matplotlib.pyplot as plt

def equacao_diferencial(y, x):
    return x - y

def metodo_euler(eq_func, x0, y0, h, n):
    resultados = []
    for i in range(n + 1):
        resultados.append((x0, y0))
        y0 = y0 + h * eq_func(y0, x0)
        x0 = x0 + h
    return resultados

# Solicitar os valores iniciais e de parâmetros ao usuário
x0 = float(input("Digite o valor inicial de x: "))
y0 = float(input("Digite o valor inicial de y: "))
h = float(input("Digite o tamanho do passo (h): "))
n = int(input("Digite o número de passos: "))

# Resolver a equação diferencial usando o método de Euler
resultados = metodo_euler(equacao_diferencial, x0, y0, h, n)
x_vals = np.array([x for x, _ in resultados])
y_vals = np.array([y for _, y in resultados])

# Plotar o gráfico
plt.plot(x_vals, y_vals, label='Solução numérica')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solução da Equação Diferencial y\' = x - y')
plt.legend()
plt.grid()
plt.show()
