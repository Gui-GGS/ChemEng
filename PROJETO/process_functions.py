import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc
import time
from matplotlib.ticker import FormatStrFormatter
import sympy as sp
##################################################################################################################################
# Define o método de Runge-Kutta de quarta ordem
def rk4(f, Y0, time_window, h):
    """
    Resolve um sistema de equações diferenciais ordinárias (EDOs) utilizando o método de Runge-Kutta de quarta ordem.

    Args:
        f: função que retorna a derivada da função Y em relação ao tempo. Deve receber dois argumentos: t e Y.
        Y0: lista ou array contendo as condições iniciais para as funções Y. Deve ter o mesmo comprimento que o número de equações em f.
        time_window: lista ou tupla contendo o tempo inicial (t0) e o final (tf).
        h: tamanho do passo.

    Returns:
        Uma tupla contendo dois arrays: o primeiro com o tempo (t) e o segundo com as soluções do sistema de EDOs para cada tempo (Y).
    """

    # Calcula o número total de passos a serem dados
    t0, tf = time_window
    n = int((tf-t0)/h)

    # Cria arrays para armazenar os resultados
    t = np.zeros(n+1)
    Y = np.zeros((n+1, len(Y0)))
    Y[0] = Y0
    t[0] = t0
    
    # Aplica o método de Runge-Kutta de quarta ordem
    for i in range(n):
        k1 = h*np.array(f(t[i], Y[i]))
        k2 = h*np.array(f(t[i] + h/2, Y[i] + k1/2))
        k3 = h*np.array(f(t[i] + h/2, Y[i] + k2/2))
        k4 = h*np.array(f(t[i] + h, Y[i] + k3))
        Y[i+1] = Y[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = t[i] + h
    
    # Retorna os resultados como uma tupla de arrays
    return t, Y
##################################################################################################################################
# Plota os resultados
def plot_doubleY(t, Y, x_label="Posição (m)", y_label="Temperatura (°F)"):
    plt.plot(t, Y[:,0], label='x(t)')
    plt.plot(t, Y[:,1], label='y(t)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
##################################################################################################################################
# Exemplo para teste RK4
"""
# Define as EDOs
def system_of_odes(t, Y):
    x, y = Y
    dx_dt = (0.8*1.25*np.pi/(9820*.425))*(y-x)
    dy_dt = (0.8*1.25*np.pi/(6330*.44))*(x-y)
    return [dx_dt, dy_dt]
# Define as condições iniciais e as configurações da simulação
time_range = [0, 1400]
Y0 = [60, 170]
# Resolve as EDOs com o método de Runge-Kutta de quarta ordem
t, Y = rk4(system_of_odes, Y0, time_range, 100)
plot_doubleY(t, Y)
"""
##################################################################################################################################
#cstr simples isotérmico A -> B
def cstr(n_reatores, tag=1, metodo=0, grafico=1):
    """
    Resolve o problema de um sistema de reatores contínuos em série, onde o reagente A é consumido em cada reator.

    Parâmetros:
    n_reatores (int): número de reatores na série.
    tag (int): se 0, os valores de entrada devem ser inseridos pelo usuário. Se 1, valores pré-definidos serão usados.
    metodo (int): método a ser usado para resolver o sistema de equações. Se 0, fsolve do scipy será usado. Se 1, a inversa da matriz A será usada.
    grafico (int): se 1, um gráfico dos resultados será plotado. Se 0, uma tupla com os resultados será retornada.

    Retorna:
    Se grafico=1, nenhum valor é retornado. Se grafico=0, uma tupla com dois elementos é retornada. O primeiro elemento é uma lista com o número dos reatores, o segundo é uma lista com as concentrações de A em cada reator.

    """
    print("Digite os valores das vazões positivas para entrada e negativas para saída de cada reator!", "\n", 
          "Os vetores de vazão (Q_n) devem ser compostos em cada posição pela soma das vazões associadas a concentração", "\n",
          "do reagente A em cada tanque [c_a1, c_a2, ..., c_an], exemplo: o vetor vazão do tanque 1 é Q_1 =[-10, 0, ..., 0]")
    A=[]
    if tag == 0:
        for i in range(1, n_reatores+1):
            Q=input("Insira Q_"+ str(i)+ " com as "+str(n_reatores)+" entradas do vetor separadas por espaço: ")
            A.append([int(vaz) for vaz in Q.split(' ')])
        k=[int(const) for const in input("Insira as constantes de reação separadas por espaço: ").split(' ')]
        V=[int(vol) for vol in input("Insira os volumes de cada um dos reatores separados por espaço: ").split(' ')] 
        entrada = int(input("Qual o valor da entrada multiplicada pela vazão? "))       
    else:
        A=[[-10, 0, 0, 0], [10, -15, 5, 0], [0, 15, -18, 3], [0, 0, 13, -13]]
        k=[0.075, 0.15, 0.4, 0.1]
        V=[25, 75, 100, 25]
        entrada = 10 
    start = time.time()     
    r = [-k[i]*V[i] for i in range(n_reatores)]
    for i in range(n_reatores):
        A[i][i] += r[i]
    b = np.transpose([[-entrada] + [0 for k in range(n_reatores-1)]])
    
    if metodo==0:
        def equa(x, *data):
            A_matrix, b_matrix = data
            guess = np.transpose([x])
            return np.transpose(np.matmul(A_matrix, guess) - b_matrix)[0]
        dados = (A, b)
        roots = sc.fsolve(equa, [1 for i in range(n_reatores)], args=dados)
    else:
        roots = np.transpose(np.matmul(np.linalg.inv(A), b))[0]
    for i in range(n_reatores):
        print("C_a{0} = {1:.5f} mol/L".format(i+1, roots[i]))
        if i==(n_reatores-1):
            end = time.time()
            elapsed = end - start
            print(f"Elapsed time: {elapsed:.5f} seconds")
    if grafico==1:
        fig, ax = plt.subplots()
        ax.set_xlabel('Reator')
        plt.xticks(range(1, n_reatores+1))
        ax.set_ylabel('[C] (mol/L)')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title('Concentração do Reagente A pelo processo')
        plt.scatter(range(1, n_reatores+1), roots)
        plt.show()
    else:
        return (range(1, n_reatores+1), roots)

##################################################################################################################################

