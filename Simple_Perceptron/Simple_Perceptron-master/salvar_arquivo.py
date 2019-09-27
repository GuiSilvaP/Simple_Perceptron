# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:28:10 2019
configurações para leitura dos arquivos que contem os dados
@author: Guilherme
"""

    stream = open('Configuracoes_iniciais.yml', 'r', encoding='utf-8').read() #lendo as config. iniciais
    configurations = yaml.load(stream=stream, Loader=yaml.FullLoader)    	 
    taxa_aprendizado = configurations['taxa_aprendizado']
    numero_epocas = configurations['numero_epocas']  	 
    print("\n taxa de aprendizado",taxa_aprendizado,"\n Epocas utilizadas",numero_epocas)
    
    print("\n")
    print("******************** PERCEPTRON SIMPLES ********************")
    print("\n Escolha o tipo de base a ser executado \n")
    print("\n1 - Porta Logica OR \n2 - PORTA Logica AND \n3 - Iris")
    opcao = int(input()) 
	 
    if opcao == 1:      	 
        entrada_arquivo = []             	 
        with open('OR.csv', newline='') as csvfile: # lendo o arquivo OR
            arquivo = csv.reader(csvfile, delimiter=',')            	 
            for linha in arquivo:
                    entrada_arquivo.append(list(map(int, linha)))
        entrada = np.array(entrada_arquivo)
        print("\n entrada lida do arquivo atribuido OR \n",entrada)

        inicializa_bias(entrada)
        add_bias_entrada(bias, entrada)

        num_colunas = entrada.shape[1]
        inicializa_pesos(num_colunas)

        realiza_treinamento(entrada, numero_epocas)
    print("************************************************************")
    if opcao == 2:      	 
        entrada_arquivo = []             	 
        with open('AND.csv', newline='') as csvfile: # lendo o arquivo OR
            arquivo = csv.reader(csvfile, delimiter=',')            	 
            for linha in arquivo:
                    entrada_arquivo.append(list(map(int, linha)))
        entrada = np.array(entrada_arquivo)
        print("\n entrada lida do arquivo atribuido AND \n",entrada)
    
        inicializa_bias(entrada)
        add_bias_entrada(bias, entrada)
    
        num_colunas = entrada.shape[1]
        inicializa_pesos(num_colunas)
    
        realiza_treinamento(entrada, numero_epocas)
    print("************************************************************")  
