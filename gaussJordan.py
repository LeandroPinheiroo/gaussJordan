#######################################################################################################################
## João Paulo de Souza                                                                                               ##
## Leandro Souza Pinheiro                                                                                            ##
## Trabalho Prático 3 -> Vetores e  Matrizes                                                                         ##
#######################################################################################################################
import numpy as np


# Metodo de Gauss Jordan
# Adaptado da video aula -> https://www.youtube.com/watch?v=xOLJMKGNivU
def gauss(matrix,vector):
    n = len(vector)
    for k in range(n):
        #Realiza o pivoteamento
        for i in range(k+1,n):
            if np.fabs(matrix[i][k]) > np.fabs(matrix[k][k]):
                for j in range(k,n):
                    matrix[k][j],matrix[i][j] = matrix[i][j],matrix[k][j]
                vector[k],vector[i] = vector[i],vector[k]
                break
        #Divide pelo pivo da linha atual
        pivot = matrix[k][k]
        for j in range(k,n):
            matrix[k][j] /= pivot
        vector[k] /= pivot
        #Realiza a eliminação para resolver o sistema
        for i in range(n):
            if i != k and matrix[i][k] != 0:
                factor = matrix[i][k]
                for j in range(k,n):
                    matrix[i][j] -= factor * matrix[k][j]
                vector[i] -= factor * vector[k]
    #Retorna a resposta
    return vector


if __name__ == '__main__':
    #Define matriz 
    matrix = [
        [ 3.0 , 5.0 , 0.0 , 1.0 ],
        [ 5.0 , 2.0 , 3.0 , 2.0 ],
        [ 1.0 , 30.0 , 0.0 , 1.0 ],
        [ 5.0 , 25.0 , -6.0 , -5.0 ]
    ]
    #Define vetor
    vector = [0,-2,-7,6]

    #calculo com o Numpy
    resultNp = np.linalg.inv(matrix).dot(vector)
    print("Solução pelo Numpy: ")
    print(resultNp)

    #calculo com o metodo de gauss implementado
    resultGauss = gauss(matrix,vector)
    print("Solução pela eliminação de Gauss: ")
    print(resultGauss)