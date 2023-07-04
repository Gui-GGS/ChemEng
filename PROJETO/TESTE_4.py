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

def equa4(c, data):
    x1, x2, x3, x4, L = c
    x_ = [x1, x2, x3, x4]
    F_, k, z = data
    y = [k[0]*x1, k[1]*x2, k[2]*x3, k[3]*x4]
    y1, y2, y3, y4 = y
    z1, z2, z3, z4 = z
    V = F_ - L
    return [V*y1 + L*x1 - F_*z1, V*y2 + L*x2 - F_*z2, V*y3 + L*x3 - F_*z3, V*y4 + L*x4 - F_*z4, sum(x_)**2 - 1]

def flash(equations, f_solve=1, numerico=1, tag=0):
    if tag==0:
        # usa dados padrões 
        dados = [F_, K, Z]
    else:
        # usa dados fornecdos pelo usuário
        dados = [int(const) for const in input("Insira F, V, A1, B1, C1, k separados por espaço: ").split(' ')]
    if f_solve==1:
        # resolve pelo solver
        s = time.time()
        roots = sc.fsolve(equations, [1 for i in range(1)], args=dados)
        e = time.time()
    else:
        # calcula Jacobiano analíticamente
        s = time.time()
        roots = get_J3(dados, _type=numerico, flag=tag)
        e = time.time()
    print("\n", "Tipo: {0}, Método: {1}".format("fsolve" if f_solve==1 else "newton-raphson", "Analítico" if numerico==0 else "Numérico"))
    seconds = e - s
    print(f"Tempo parcial: {seconds:.5f} segundos")
    y = np.zeros(4)
    for i in range(len(roots)):
        compound = ["Propano", "n-Butano", "n-Pentano", "n-Hexano"]
        if i <= 3:
            print(compound[i])
            print("x{0} = {1:.3f} %".format(i+1, roots[i]*100))
            y[i] = roots[i]*K[i]
            print("y{0} = {1:.3f} %".format(i+1, y[i]*100))
            print("\n")
        else:
            vapor = F_ - roots[-1]
            print("V = {0:.5f} mols/h".format(vapor), "\n", "L = {0:.5f} mols/h".format(roots[-1]), sep="")
    # print(sum(roots[0:4]), vapor + roots[-1], sum(y))
    


def newton_raphson_system(F, J, _type=1, tag=0, method=0, maxiter=100, tol=1e-6, x_0=[1, 1, 1, 1, 1]):
    """
    Solves a system of nonlinear equations F(x) = 0 using the Newton-Raphson method.

    Parameters:
        F (function): A function that takes a vector x as input and returns a vector of function values.
        J (function): A function that takes a vector x as input and returns a Jacobian matrix.
        x0 (ndarray): An initial guess for the solution.
        maxiter (int): Maximum number of iterations.
        tol (float): Tolerance for stopping criterion.
        method (int): 0 for solver, 1 for matrix inversion

    Returns:
        x (ndarray): The solution to the system of equations.
    """
    if tag==1:
        # pede dados do usuário
        x0_str = input("Enter initial guess separated by commas: ")
        x0 = np.array([float(x) for x in x0_str.split(',')])
    else:
        # resolve o sistema padrão
        x0 = np.array(x_0)
    x = [x0.copy()]
    for i in range(maxiter):
        if i != 0:
            x = np.transpose(x)
        f = F(*x[0])
        if _type == 0:
            Jx = J(x[0]).astype('float64')
        elif _type == 1:
            Jx = J(F, x[0]).astype('float64')
        x_mult = np.transpose(x)
        f_mult = np.transpose([f])
        B = -1*f_mult + np.matmul(Jx, x_mult)
        if method == 0:
            dx = np.linalg.solve(Jx, B)
        elif method == 1:
            Jx_inv = np.linalg.inv(Jx)
            dx = np.matmul(Jx_inv, B)
        x = dx
        if np.linalg.norm(f) < tol:
            x = np.transpose(x)
            return x[0]
    raise ValueError("Newton-Raphson method did not converge")


def get_J3(dados, _type=1, flag=0):
    # define variables and create function for F
    def F(*x): 
        x1, x2, x3, x4, L = x
        x_ = [x1, x2, x3, x4]
        F_, k, z = dados
        y = [k[0]*x1, k[1]*x2, k[2]*x3, k[3]*x4]
        y1, y2, y3, y4 = y
        z1, z2, z3, z4 = z
        V = F_ - L
        return [V*y1 + L*x1 - F_*z1, V*y2 + L*x2 - F_*z2, V*y3 + L*x3 - F_*z3, V*y4 + L*x4 - F_*z4, sum(x_)**2 - 1]
    
    def J_func(F, x, eps=1e-6):
        """
        Computes the Jacobian matrix numerically for a system of n equations and n variables using finite differences.

        Parameters:
            F (function): A function that takes a vector x as input and returns a vector of function values.
            x (ndarray): A vector of current values for the variables.
            eps (float): A small value for the finite difference approximation.

        Returns:
            J (ndarray): The Jacobian matrix.
        """
        n = len(x)
        Jacob = np.zeros((n, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1
            Jacob[:, j] = (np.transpose(F(*(x + eps * e))) - np.transpose(F(*x))) / eps
            e[j] = 0
        return Jacob
    # solve system using Newton-Raphson method
    return newton_raphson_system(F, J_func, _type=_type)

flash(equa4, f_solve=0)


