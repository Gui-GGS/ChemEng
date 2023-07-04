import numpy as np
import pandas as pd
import scipy.optimize as sc
import time


def antoine(a: float, b: float, c: float, t: float):
    """
        Calcula Psat dado os coeficientes de Antoine da equação no formato log(P_sat) = A - B/(C+T).

        Args:
        a (float): A.

        b (float): B.

        c (float): C.

        t (float): Temperatura (°C).

        Retorna:
        P_sat (float): Psat em kPa
    """
    return (10**(a - (b/(c+t))))*0.133322

coef = pd.read_excel("237960_PROJETO_1.xlsm", "Questão4", usecols="A:D", nrows=4)
startup = [1e5, 689.5, 366.5]
Z = [.1, .2, .3, .4]
F_, P_sys, T_sys = startup
P_sat = np.zeros(len(Z))
K = np.zeros(len(Z))
for i in range(len(Z)):
    P_sat[i] = (antoine(coef["A"][i], coef["B"][i], coef["C"][i], T_sys - 273.15))
for i in range(len(Z)):
    K[i] = P_sat[i]/P_sys

def Rachford(psi, dados):
    z, k = dados
    return sum([(z[i]*(1-k[i]))/(1+psi*(k[i]-1)) for i in range(len(z))])

Psi = sc.fsolve(Rachford, 1, args=[Z, K])
x = np.zeros(4)
y = np.zeros(4)
for i in range(len(Z)+1):
    compound = ["Propano", "n-Butano", "n-Pentano", "n-Hexano"]
    if i <= 3:
        print(compound[i])
        x[i] = Z[i]/(1 + Psi*(K[i] - 1))
        print("x{0} = {1:.3f} %".format(i+1, x[i]*100))
        y[i] = x[i]*K[i]
        print("y{0} = {1:.3f} %".format(i+1, y[i]*100))
        print("\n")
    else:
        vapor = F_*Psi[0]
        Liq = F_ - vapor
        print("V = {0:.5f} mols/h".format(vapor), "\n", "L = {0:.5f} mols/h".format(Liq), sep="")

