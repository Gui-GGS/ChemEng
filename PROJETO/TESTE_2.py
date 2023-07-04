import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc
import time
from matplotlib.ticker import FormatStrFormatter
import sympy as sp

def equa2(c, data):
    A, B, C, D = c
    F, V, A1, B1, C1, D1 = data
    return [F*(A1-A)-0.855*A*B*V, F*(B1-B)-0.855*A*B*V, F*(C1-C)+0.855*A*B*V, F*(D1-D)+0.855*A*B*V]


def cstr2(f_solve=1, numerico=1, tag=0):
    if tag==0:
        # usa dados padrões 
        dados = [5, 40, 0.7, 0.4, 0., 0.]
    else:
        # usa dados fornecdos pelo usuário
        dados = [int(const) for const in input("Insira F, V, A1, B1, C1, D1 separados por espaço: ").split(' ')]
    if f_solve==1:
        # resolve pelo solver
        s = time.time()
        roots = sc.fsolve(equa2, [1 for i in range(4)], args=dados)
        e = time.time()
    else:
        # calcula Jacobiano analíticamente
        s = time.time()
        roots = get_J(dados, _type=numerico, flag=tag)
        e = time.time()
    print("\n", "Tipo: {0}, Método: {1}".format("fsolve" if f_solve==1 else "newton-raphson", "Analítico" if numerico==0 else "Numérico"))
    seconds = e - s
    print(f"Tempo parcial: {seconds:.5f} segundos")
    for i in range(4):
        letters = ["a", "b", "c", "d"]
        print("C_{0} = {1:.5f} mol/L".format(letters[i], roots[i]))


def newton_raphson_system(F, J, _type=1, tag=0, method=0, maxiter=100, tol=1e-6, x_0=[1, 1, 1, 1]):
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
    start_time = time.time()
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
            end_time = time.time()
            elapsed_time = end_time - start_time
            # print(f"Elapsed time: {elapsed_time:.5f} seconds")
            return x[0]
    raise ValueError("Newton-Raphson method did not converge")


def get_J(dados_, _type=1, flag=0):
    # get input equations from user
    if flag==1:
        eqns_str = input("Enter system of equations separated by commas (use x0, x1, ..., xn for variables): ")
        eqns = [sp.sympify(eqn) for eqn in eqns_str.split(',')]
    else:
        F, V, A1, B1, C1, D1 = dados_
        eqns_str = "{4}*({5}-{0})-0.855*{0}*{1}*{9}, {4}*({6}-{1})-0.855*{0}*{1}*{9}, {4}*({7}-{2})+0.855*{0}*{1}*{9}, {4}*({8}-{3})+0.855*{0}*{1}*{9}".format("x0", "x1", "x2", "x3", F, A1, B1, C1, D1, V)
        eqns = [sp.sympify(eqn) for eqn in eqns_str.split(',')]
    # define variables and create function for F
    vars = sp.symbols('x0:%d' % len(eqns))
    F = sp.lambdify(vars, eqns)
    if _type == 0:
        # compute Jacobian matrix using SymPy
        J = sp.Matrix.zeros(len(eqns), len(eqns))
        for i in range(len(eqns)):
            for j in range(len(eqns)):
                J[i, j] = sp.diff(eqns[i], vars[j]).evalf()
        def J_func(x):
            return np.array([[J[i,j].subs(zip(vars, x)).evalf() for j in range(len(eqns))] for i in range(len(eqns))])
    else:
        print("numerico Jfunc")
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
    # print("Solution: ")
    # for i in range(len(x)):
    #     print('x{0}: {1:.4f}'.format(i, x[i]))

cstr2(f_solve=0)
cstr2(f_solve=0, numerico=0)
cstr2(f_solve=1)