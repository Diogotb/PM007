import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Função que descreve o sistema de equações diferenciais SIR
def SIR_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Parâmetros do modelo
beta = 0.3   # Taxa de contágio
gamma = 0.1  # Taxa de recuperação

# Condições iniciais
S0 = 0.99   # Suscetíveis
I0 = 0.01   # Infectados
R0 = 0.0    # Recuperados

# Intervalo de tempo
t_span = (0, 200)

# Resolver o sistema de equações diferenciais
solution = solve_ivp(SIR_model, t_span, [S0, I0, R0], args=(beta, gamma), t_eval=np.linspace(0, 200, 1000))

# Plotar os resultados
plt.plot(solution.t, solution.y[0], label='Suscetíveis')
plt.plot(solution.t, solution.y[1], label='Infectados')
plt.plot(solution.t, solution.y[2], label='Recuperados')
plt.xlabel('Tempo')
plt.ylabel('Proporção da População')
plt.title('Modelo SIR - Propagação de uma Doença')
plt.legend()
plt.grid()
plt.show()
