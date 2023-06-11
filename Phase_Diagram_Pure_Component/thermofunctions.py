from typing import Callable, Any
import ast
import sys
import sympy as sp
# usadas em tg01
import numpy as np
import functools
import matplotlib.pyplot as plt
########################################################################################################################


# calculando número estimado de iterações
def num_ite(a, b, err):
    n = sp.log((b - a) / err) / sp.log(2)
    return int(round(n, 0))


# teorema do valor médio corrigido para o código
def validade_intervalo(f, x0, x1):
    return f(x0) * f(x1) <= 0.0


# funções do cálculo numérico
def bisse(f, interval, clean=True, tol=1e-10):
    """
    param f: função sob análise
    param interval: sob intervalo do tipo [a, b] | a<b
    param tol: tolerância da precisão (min: 10^-11)
    """
    error_list = []
    # extraindo intervalo
    x0, x1 = interval[0], interval[1]
    # conferindo validade do intervalo
    if not validade_intervalo(f, x0, x1):
        return
    # iterações necessárias para encontrar uma raiz no intervalo com a tol determinada
    n = num_ite(x0, x1, tol)
    counter = 1
    # iterando dentro da margem de erro
    while True:
        # aproximação da raiz (ponto médio)
        root_approx = x0 + ((x1 - x0) / 2.0)
        # avaliando y no ponto médio
        y = f(root_approx)
        error_list.append(y)
        # conferindo condição de tol
        if (-tol < y < tol and abs(x1-x0) < tol) or counter > 100000:
            # conferindo o número de iterações
            if not clean:
                return [root_approx, counter, y, abs(error_list[-1]-error_list[-2])]
            else:
                return root_approx
        # conferindo o prox seguimento com a bissetriz
        if validade_intervalo(f, x0, root_approx):
            x1 = root_approx
        else:
            x0 = root_approx
        counter += 1


# outros métodos
def falsa_pos(f, interval, clean=True, tol=1e-10):
    """
    param f: função sob análise
    param interval: sob intervalo do tipo [a, b] | a<b
    param tol: tolerância da precisão (min: 10^-11)
    """
    error_list = []
    # extraindo intervalo
    x0, x1 = interval[0], interval[1]
    # conferindo validade do intervalo
    if not validade_intervalo(f, x0, x1):
        return
    # iterações necessárias para encontrar uma raiz no intervalo com a tol determinada
    n = num_ite(x0, x1, tol)
    counter = 1
    # iterando dentro da margem de erro
    while True:
        # aproximação da raiz (ponto médio)
        root_approx = (x0*f(x1) - x1*f(x0)) / (f(x1)-f(x0))
        # avaliando y no ponto médio
        y = f(root_approx)
        error_list.append(y)
        # conferindo condição de tol
        if (-tol < y < tol and abs(x1-x0)) or counter > 1000:
            # conferindo o número de iterações
            if not clean:
                return [root_approx, counter, y, abs(error_list[-1]-error_list[-2])]
            else:
                # retornando a raiz
                return root_approx
        # conferindo o prox seguimento com a bissetriz
        if validade_intervalo(f, x0, root_approx):
            x1 = root_approx
        else:
            x0 = root_approx
        counter += 1


# pegar n questao
def get_ans(prompt="? "):
    try:
        result = ast.literal_eval(input(prompt))
        if not isinstance(result, (float, int)):
            raise ValueError
        if result == 0:
            raise KeyboardInterrupt
        return result
    except ValueError:
        print("Essa não é uma entrada válida, tente novamente:")
        return get_ans(prompt)
    except KeyboardInterrupt:
        print('Parada forçada pelo usuário')
        sys.exit(1)

########################################################################################################################


# antoine equation
# returns Psat to given parameters, constants a, b, d and temperature t
def antoine(t):
    a = 6.80267
    b = 656.4028
    c = 273.15-255.99
    return (10**(a - (b/(t-c))))*0.00133322


########################################################################################################################
# eccentric factor function
eccentric: Callable[[Any], float] = lambda w: \
    0.37464+1.54226*w-0.26992*w**2


########################################################################################################################
# Combined gases Law
# return volume in L/mol
# R in L.bar/K.mol
def law(v, p, t, equation=1, ecc=0.099, tc=305.32, pc=48.72, r=8.31451e-2):
    # 1 - vdw
    # 2 - rk
    # 3 - pr
    tr = t/tc
    pr = p/pc
    if equation == 1:
        a = (27*(r**2)*(tc**2))/(64*pc)
        # print(a)
        b = (r*tc)/(8*pc)
        # print(b)
        u = 0
        w = 0
        return ((v ** 2) + u * b * v + w * b ** 2)*(v - b)*p - ((v ** 2) + u * b * v + w * b ** 2)*(r * t) + a*(v - b)
    if equation == 2:
        a = (0.42748*(r**2)*(tc**2.5))/(pc*(t**0.5))
        # print("a", a*(t**0.5))
        b = (0.08664*r*tc)/pc
        # print("b", b)
        u = 1
        w = 0
        return ((v ** 2) + u * b * v + w * b ** 2)*(v - b)*p - ((v ** 2) + u * b * v + w * b ** 2)*(r * t) + a*(v - b)
    if equation == 3:
        a = (0.45724*(r**2)*(tc**2.5)/pc)*(1+eccentric(ecc)*(1-tr**0.5))**2
        b = 0.07780*r*tc/pc
        u = 2
        w = -1
        return ((v ** 2) + u * b * v + w * b ** 2)*(v - b)*p - ((v ** 2) + u * b * v + w * b ** 2)*(r * t) + a*(v - b)


########################################################################################################################
# finding roots
def cubicPoints(f, passo=0.002, intervalo=[0, 200]):
    a0 = intervalo[0]
    b0 = intervalo[1]
    return list(filter(None, [bisse(f, [i*passo, (i+1)*passo]) for i in range(round(abs(b0-a0)/passo))]))


def cubicPoints2(f, passo=0.002, intervalo=[0, 200]):
    a0 = intervalo[0]
    b0 = intervalo[1]
    return list(filter(None, [falsa_pos(f, [i*passo, (i+1)*passo]) for i in range(round(abs(b0-a0)/passo))]))
