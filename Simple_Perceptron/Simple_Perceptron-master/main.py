'''
IFCE Maracanaú
Nome : Guilherme Pessoa

---------Perceptron Simples-------------------
'''
import matplotlib
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
import yaml
import csv

acuracia = []

base_train = np.array([])
base_test = np.array([])

w = np.array([])
bias = np.array([])

#entrada = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
#entrada = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])


def inicializa_pesos(_entrada):
   global w
   w = np.array([])

   for i in range(1, _entrada):
       w = np.append(w, [0.5])
   print("w final: {}".format(w))


def inicializa_bias(_entrada):
   global bias
   bias = np.array([])
   bias = np.array([-1] * len(_entrada))


def add_bias_entrada(_bias, _np_entrada):
   # global y_desejados
   global entrada
   entrada = np.column_stack((_bias, _np_entrada))
   # y_desejados = entrada_com_bias_completa[:, -1]  # for last column
   # entrada_com_bias = entrada_com_bias_completa[:, :-1]  # for all but last column


def func_deg(u):  # retorna o y
   if u > 0:
       return 1
   elif u <= 0:
       return 0


def verifica_erro(y_calculado, y_desejado):  # verifica se y calculado = y desejado
   return y_desejado - y_calculado


def atualiza_pesos(erro, entrada_com_bias):
   global w
   global taxa_aprendizado

   x_novo = taxa_aprendizado * erro * entrada_com_bias
   w = np.sum([x_novo, w], axis=0)




def plot_data(inputs, targets, weights):
    global bias
    b, w1, w2 = weights
    #w_norm = preprocessing.normalize(weights, norm='l1')
    #weights = w_norm
    #weights = (weights / weights.max(axis=0))
    
    b, w1, w2 = weights
    plt.figure(figsize=(10, 5))
  
    x = -b / w1
    y = -b/ w2
    
    d = y
    c = -y / x
    
    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets)
    ponto_x, ponto_y = [0, 0],[0, 0]#[0, x], [y, 0]
    plt.plot(ponto_x, ponto_y)
    plt.show()

#print("teste_base: \n{}".format(base_train))
#print("tra_base: \n{}".format(base_test))

#plt.scatter(ponto_x, ponto_y)
#plt.show()


def realiza_treinamento(_entrada, _numero_epocas):
   global base_train
   global base_test
   global w

   for epoch in range(1, _numero_epocas):
       # A época inicia c/ erro zero
       erro_epoch = 0

       # Os dados da entrada são embaralhados
       np.random.shuffle(_entrada)
       # Pego 80% dos dados p/ treinamento e 20% p/ teste
       base_train, base_test = train_test_split(_entrada, test_size=0.2)

       # Separo a entrada do valor desejado
       entrada_sem_y = base_train[:, :-1]
       y_desejado = base_train[:, -1]

       for i in range(0, len(base_train)):
           entrada_atual = entrada_sem_y[i]
           u = sum(np.multiply(entrada_atual, w))
           y_calculado = func_deg(u)
           erro = verifica_erro(y_calculado, y_desejado[i])

           atualiza_pesos(erro, entrada_atual)
           if erro != 0:
               erro_epoch = erro_epoch + 1

       if erro_epoch > 0:
           pass
       else:
           print("\nÉPOCAS UTILIZADAS: {}".format(epoch))
           break

       realiza_teste(base_test, w)
   # TESTE
   print("TREINAMENTO: \n{}\nTESTE: \n{}\nW: \n{}".format(base_train, base_test, w))




def realiza_teste(_teste_entrada, _pesos):
   acertos = 0
   erros = 0
   
   tx_acerto = 0
   for i in range(0, len(_teste_entrada)):
       entrada_sem_y_teste = _teste_entrada[:, :-1]
       y_desejado_teste = _teste_entrada[:, -1]

       entrada_atual_teste = entrada_sem_y_teste[i]
       y_desejado_atual_teste = y_desejado_teste[[i]]

       #u = sum(np.multiply(entrada_atual_teste, _pesos))

       #a = entrada_atual_teste.reshape((-1, 1))
       _pesos = _pesos.reshape((-1, 1))

       u = sum(np.dot(entrada_atual_teste, _pesos))

       y_calculado = func_deg(u)

       erro = verifica_erro(y_calculado, y_desejado_atual_teste)

       if erro == 0:
           acertos = acertos + 1
           print("\nA entrada {} foi classificada corretamente!".format(entrada_atual_teste))
           print("Classificado como: {}".format(y_calculado))
           print("Deveria ser: {}".format(y_desejado_atual_teste))
       else:
           erros = erros + 1
           print("\n")
           print("Erro ao classificar a entrada {}".format(entrada_atual_teste))
           print("Classificado como: {}".format(y_calculado))
           print("Deveria ser: {}".format(y_desejado_atual_teste))
   print("\nAcertos: {}\nErros: {}".format(acertos, erros))
   tx_acerto = acertos/(acertos+erros)
   acuracia.append(tx_acerto)


if __name__ == '__main__': #Inicializando os dados lidos do arquivo
               
    stream = open('Configuracoes_iniciais.yml', 'r', encoding='utf-8').read() #lendo as config. iniciais
    configurations = yaml.load(stream=stream, Loader=yaml.FullLoader)    	 
    taxa_aprendizado = configurations['taxa_aprendizado']
    numero_epocas = configurations['numero_epocas']  
    realizacao = configurations['realizacao']	 
    print("\n taxa de aprendizado",taxa_aprendizado,"\n Epocas utilizadas",numero_epocas)
    
    print("\n") #gerando um imput para o usuario escolher qual base a ser testada
    print("*************** PERCEPTRON SIMPLES **************")
    print("\n Escolha o tipo de base a ser executado \n")
    print("\n1 - Porta Logica OR \n2 - PORTA Logica AND \n3 - Iris")
    opcao = int(input()) 
    
    for i in range(1,realizacao):
        if opcao == 1: # 1° opcao carrega o database da Porta logica OR 	 
            entrada_arquivo = []             	 
            with open('OR.csv', newline='') as csvfile: # lendo o arquivo OR
                arquivo = csv.reader(csvfile, delimiter=',')            	 
                for linha in arquivo:
                        entrada_arquivo.append(list(map(float, linha)))
            entrada = np.array(entrada_arquivo)
            print("\n entrada lida do arquivo atribuido OR \n",entrada)
            
            numero_linhas_entrada = len(entrada)
            inicializa_bias(entrada)
            add_bias_entrada(bias, entrada)
        
            num_colunas = entrada.shape[1]
            inicializa_pesos(num_colunas)
        
            realiza_treinamento(entrada, numero_epocas)    
    
            x_entrada = np.delete(entrada, 0, 1)
            y_entrada = entrada[:, -1]
        
            print("x_entrada: {}".format(x_entrada))
            print("y_entrada: {}".format(y_entrada))
            plot_data(x_entrada, y_entrada, w)
            
        print("************************************************************")    
        
        if opcao == 2: # 2° opcao carrega o database da Porta logica AND 	 
            entrada_arquivo = []             	 
            with open('AND.csv', newline='') as csvfile: # lendo o arquivo OR
                arquivo = csv.reader(csvfile, delimiter=',')            	 
                for linha in arquivo:
                        entrada_arquivo.append(list(map(float, linha)))
            entrada = np.array(entrada_arquivo)
            print("\n entrada lida do arquivo atribuido AND \n",entrada)
            
            numero_linhas_entrada = len(entrada)
            inicializa_bias(entrada)
            add_bias_entrada(bias, entrada)
        
            num_colunas = entrada.shape[1]
            inicializa_pesos(num_colunas)
        
            realiza_treinamento(entrada, numero_epocas)    
    
            x_entrada = np.delete(entrada, 0, 1)
            y_entrada = entrada[:, -1]
        
            print("x_entrada: {}".format(x_entrada))
            print("y_entrada: {}".format(y_entrada))
            plot_data(x_entrada, y_entrada, w)
    
        print("************************************************************")
        
        if opcao == 3: # 3° opcao carrega o database da Porta logica AND 	 
            
            iris = datasets.load_iris()
            X = iris.data[:, :2]  # we only take the first two features.
            y = np.array(iris.target)
    
            y = np.where(y == 1, 2, y)
            # print(y)
      
            # adiciona o y_desejado ao final da entrada
            entrada = np.column_stack((X, y))
            numero_linhas_entrada = len(entrada)
            
            # add bias no inicio da entrada
            inicializa_bias(entrada)
            add_bias_entrada(bias, entrada)
    
            num_colunas = entrada.shape[1]
            inicializa_pesos(num_colunas)
        
            #NORMALIZAÇÃO
            min_max_scaler = MinMaxScaler()
            X_train_norm = min_max_scaler.fit_transform(entrada)
        
            num_colunas_X_train_norm = X_train_norm.shape[1]
        
            realiza_treinamento(X_train_norm, numero_epocas)
        
            x_entrada = np.delete(X_train_norm, 0, 1)
            y_entrada = X_train_norm[:, -1]
            plot_data(x_entrada, y_entrada, w)
            #Mostra a acuracia da base de dados Iris
            print("Acurácia: {}".format(np.mean(acuracia)))
            
    print ("Treinamento Finalizado")
    print ("Numero de realizacoes :", realizacao)
    

