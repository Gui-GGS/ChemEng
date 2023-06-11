from thermofunctions import *
#######################################################################################################################
# creating data matrix



def binodal():
    eq = int(get_ans("Qual equação quer rodar? 1 - Van der Waals; 2 - Redlich-Kwong ou 3 - Peng-Robson "))
    pressoes = []
    temperaturas = []
    volumesL = []
    volumesV = []
    pontos = int(get_ans("Quantos pontos deve ter a curva? (recomendado: 40) "))
    for i in range(pontos):
        if eq == 1:
            # t_start min 130, recommended 155
            t_start, t_end = 155, 281
            # t_end max 281
        if eq == 2:
            # t_start min 130, recommended 155
            t_start, t_end = 155, 265
            # t_end max 265
        if eq == 3:
            # t_start min 145
            t_start, t_end = 145, 166
            # t_end max 166
        psat = antoine(t_start+i*((t_end-t_start)/(pontos-1)), a=1, b=1, c=1)
        pressoes.append(psat)
        temperaturas.append(t_start+i*((t_end-t_start)/(pontos-1)))
    teste1 = True
    if teste1:
        p_t_diagram = plt.figure()
        ax = p_t_diagram.add_subplot(111)
        # Setting axes/plot title
        ax.set_title('Diagrama P x T')
        # Setting X-axis and Y-axis limits
        ax.set_xlim([round(0.9*t_start), round(1.1*t_end)])
        ax.set_ylim([round(pressoes[0]), pressoes[-1]])
        # Setting X-axis and Y-axis labels and values
        plt.plot(temperaturas, pressoes)
        ax.set_ylabel('Pressure (bar)')
        ax.set_xlabel('Temperature (K)')
        plt.show()
    # Van der waals
    # resolver divisao por zero na equação
    teste_cub = 0
    flag = 0
    for i in range(pontos):
        if eq != 3:
            vdw = functools.partial(law, p=pressoes[i], t=temperaturas[i], equation=eq, tc=1, pc=1)
        else:
            if flag == 0:
                acen = float(get_ans("Qual o fator acêntrico? "))
                flag += 1
            vdw = functools.partial(law, p=pressoes[i], t=temperaturas[i], equation=eq, ecc=acen, tc=1, pc=1)
        if temperaturas[i] > 270:
            ans = cubicPoints(vdw, passo=1e-5, intervalo=[0, 2])
        if 240 < temperaturas[i] <= 270:
            ans = cubicPoints(vdw, passo=0.002, intervalo=[0, 5])
        if temperaturas[i] <= 240:
            ans = cubicPoints(vdw, passo=0.02)
        print("psat:", pressoes[i], "temp:", temperaturas[i])
        print("raízes:", ans)
        volumesL.append(ans[0])
        volumesV.append(ans[-1])
        if teste_cub == 1:
            cubic_p_v = plt.figure()
            ax = cubic_p_v.add_subplot(111)
            # Setting axes/plot title
            psat_text = str(round(pressoes[i], 4))
            ax.set_title('Função cúbica para Psat '+psat_text)
            # Setting X-axis and Y-axis limits
            x = np.linspace(-1e10, 1e10, 1000)
            y = vdw(x)
            # Setting X-axis and Y-axis labels and values
            plt.plot(x, y)
            ax.set_ylabel('valor da função (bar)')
            ax.set_xlabel('volume (L/mol)')
            # plt.show()
    teste2 = False
    if teste2:
        p_t_diagram = plt.figure()
        ax = p_t_diagram.add_subplot(111)
        # Setting axes/plot title
        ax.set_title('Função cúbica para Psat específica')
        # Setting X-axis and Y-axis limits
        x = np.linspace(-1e10, 1e10, 1000)
        print(pressoes[15], temperaturas[15])
        vdw0 = functools.partial(law, p=pressoes[15], t=temperaturas[15], tc=1, pc=1)
        y = vdw0(x)
        # Setting X-axis and Y-axis labels and values
        # plt.xscale("log")
        plt.plot(x, y)
        ax.set_ylabel('valor da função')
        ax.set_xlabel('volume')
        plt.show()
    teste3 = True
    if teste3:
        p_v_diagram = plt.figure()
        ax = p_v_diagram.add_subplot(111)
        # Setting axes/plot title
        ax.set_title('PxV')
        # Setting X-axis and Y-axis limits
        # Setting X-axis and Y-axis labels and values
        volumesV.reverse()
        volumesL.extend(volumesV)
        pressoesV = pressoes.copy()
        pressoesV.reverse()
        pressoes.extend(pressoesV)
        # for i in range(len(volumesL)-1):
        #     if volumesL[i] < 0.147 < volumesL[i+1]:
        #         volumesL.insert(i+1, 0.147)
        #         pressoes.insert(i+1, 48.72)
        #         # print("inserido na pos:", i+1)
        plt.xscale("log")
        plt.plot(volumesL, pressoes)
        ax.set_ylabel('pressão (bar)')
        ax.set_xlabel('log volume (L/mol)')
        plt.show()

binodal()
