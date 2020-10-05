import qiskit
import numpy as np
import math
from qiskit import circuit
from qiskit import execute, Aer, QuantumRegister, QuantumCircuit
_EPS = 1e-10  # global variable used to chop very small numbers to zero
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Instruction
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.reset import Reset

def rzryrz(U):
    '''
    Lab2 - questão 1
    :param U: matriz unitária 2 x 2
    :return: [alpha, beta, gamma e delta]
            U = e^(1j * alpha) * Rz(beta) * Ry(gamma) * Rz(delta)
    '''

    # -----------------
    # Seu código aqui
    print(np.angle(U[0][0]))
    x00 =np.angle(U[0][0])
    x01 =np.angle(U[0][1]) + np.pi
    x10 =np.angle(U[1][0])
    print(x00,x01,x10)
    alpha = (x01 + x10)/2
    beta = -((x00+x01)-2*alpha)
    gamma = np.arccos(np.abs(U[0][0]))*2
    delta = (x01+beta/2-alpha)*2
    print( alpha,beta,gamma,delta)
    # -----------------

    return [alpha, beta, gamma, delta]

def operador_controlado(U):
    '''
    Lab2 - questão 2
    :param V: matriz unitária 2 x 2
    :return: circuito quântico com dois qubits aplicando o
             o operador V controlado.
    '''

    circuito = qiskit.QuantumCircuit(2)

    print(np.angle(U[0][0]))
    x00 =np.angle(U[0][0])
    x01 =np.angle(U[0][1]) + np.pi
    x10 =np.angle(U[1][0])
    print(x00,x01,x10)
    alpha = (x01 + x10)/2
    beta = -((x00+x01)-2*alpha)
    gamma = np.arccos(np.abs(U[0][0]))*2
    delta = (x01+beta/2-alpha)*2
    print( alpha,beta,gamma,delta)

    #A= z beta y gamma/2
    #b= y -gamma/2 z -(delta+beta)/2
    #c= z (delta-beta)/2
    circuito.rz((delta-beta)/2,1)
    circuito.cx(0,1)
    circuito.rz(-(delta+beta)/2,1)
    circuito.ry(-gamma/2,1)
    circuito.cx(0,1)
    circuito.ry(gamma/2,1)
    circuito.rz(beta,1)
    circuito.u3(0,alpha,0,0)
    print(circuito.draw())
    #-----------------
    # Seu código aqui
    # -----------------

    return circuito


def toffoli():
    '''
    Lab2 - questão 3
    :param n: número de controles
    :param V:
    :return: circuito quântico com n+1 qubits + n-1 qubits auxiliares
            que aplica o operador nCV.
    '''
    controles = qiskit.QuantumRegister(2)
    alvo = qiskit.QuantumRegister(1)

    circuito = qiskit.QuantumCircuit(controles, alvo)

    #------------------------
    # Seu código aqui
    circuito.h(2)
    circuito.cx(1,2)
    circuito.tdg(2)
    circuito.cx(0,2)
    circuito.t(2)
    circuito.cx(1,2)
    circuito.tdg(2)
    circuito.cx(0,2)
    circuito.t(1)
    circuito.t(2)
    circuito.cx(0,1)
    circuito.h(2)
    circuito.t(0)
    circuito.tdg(1)
    circuito.cx(0,1)
    # ------------------------

    return circuito

def inicializa_3qubits(v):
    '''
    Lab2 - questão 4
    '''

    circuito = qiskit.QuantumCircuit(3)
    
    alpha3 = np.arcsin(v[1] / (v[0]**2+ v[1]**2) ** (1/2))
    alpha4 = np.arcsin(v[3] / (v[2]**2+ v[3]**2) ** (1/2))
    alpha5 = np.arcsin(v[5] / (v[4]**2+ v[5]**2) ** (1/2))
    alpha6 = np.arcsin(v[7] / (v[6]**2+ v[7]**2) ** (1/2))
    alphaz3 = (v[0]**2+ v[1]**2) ** (1/2)
    alphaz4 = (v[2]**2+ v[3]**2) ** (1/2)
    alphaz5 = (v[4]**2+ v[5]**2) ** (1/2)
    alphaz6 = (v[6]**2+ v[7]**2) ** (1/2)
    alpha1 = 2 * np.arcsin(alphaz4 / (alphaz3**2+ alphaz4**2) **(1/2))
    alpha2 = 2 * np.arcsin(alphaz6 / (alphaz5**2+ alphaz6**2) **(1/2))
    alphaz1 = (alphaz3**2+ alphaz4**2) **(1/2)
    alphaz2 = (alphaz5**2+ alphaz6**2) **(1/2)
    alpha0 = 2 * np.arcsin(alphaz2 / (alphaz1**2+ alphaz2**2) **(1/2))
    #0, 2, 1, 6, 5, 4, 3
    circuito.ry(alpha0,2)
    circuito.cu3(alpha2,0,0,2,1)
    circuito.x(2)
    circuito.cu3(alpha1,0,0,2,1)
    circuito.cu3(alpha6,0,0,1,0)
    circuito.x(2)
    circuito.cx(1,2)
    circuito.cu3(-alpha6,0,0,2,0)
    circuito.cx(1,2)
    circuito.cu3(alpha6,0,0,2,0)
    circuito.x(1)
    circuito.cu3(alpha5,0,0,1,0)
    circuito.cx(1,2)
    circuito.cu3(-alpha5,0,0,2,0)
    circuito.cx(1,2)
    circuito.cu3(alpha5,0,0,2,0)
    circuito.x(1)
    circuito.x(2)
    circuito.cu3(alpha4,0,0,1,0)
    circuito.cx(1,2)
    circuito.cu3(-alpha4,0,0,2,0)
    circuito.cx(1,2)
    circuito.cu3(alpha4,0,0,2,0)
    circuito.x(1)
    circuito.x(2)
    circuito.cu3(alpha3,0,0,1,0)
    circuito.x(2)
    circuito.cx(1,2)
    circuito.cu3(-alpha3,0,0,2,0)
    circuito.cx(1,2)
    circuito.cu3(alpha3,0,0,2,0)
    circuito.x(1)
    circuito.x(2)

    print(circuito.draw())
    # ------------------------
    # Seu código aqui
    # ------------------------

    return circuito

def inicializa(vetor):
    '''
    Lab2 - questão 5 - opcional
    '''
    tamanho= np.log2(len(vetor))
    tamanho = int(tamanho)
    q = QuantumRegister(tamanho)
    circuito = qiskit.QuantumCircuit(q)
   
    #circuito.initialize(vetor, q1)
    print(circuito.draw())
    return circuito

if __name__ == '__main__':
    '''
    Matrix = [[0 for x in range(2)] for y in range(2)]
    Matrix[0][0] = 0;
    Matrix[0][1] = 1;
    Matrix[1][0] = 0;
    Matrix[1][1] = 1; 
    circuito = operador_controlado(Matrix)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuito, backend)
    result = job.result()
    print('Estado gerado pelo circuito: ', result.get_statevector())
    '''
    desired_vector = [
    1 / sqrt(16) * complex(0, 1),
    1 / sqrt(8) * complex(1, 0),
    1 / sqrt(16) * complex(1, 1),
    0,
    0,
    1 / sqrt(8) * complex(1, 2),
    1 / sqrt(16) * complex(1, 0),
    0]
    circuito=inicializa(desired_vector)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuito, backend)
    result = job.result()
    print('Estado gerado pelo circuito: ', result.get_statevector())
    circuito=inicializa_3qubits(desired_vector)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuito, backend)
    result = job.result()
    print('Estado gerado pelo circuito: ', result.get_statevector())