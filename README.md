Detecção de Fraudes e Viés Algorítmico no Setor Financeiro
Este repositório contém os códigos-fonte desenvolvidos para o Trabalho de Conclusão de Curso (TCC) em Ciência da Computação, focado na avaliação de modelos de Machine Learning para detecção de fraudes em transações de cartão de crédito.

O objetivo central da pesquisa é analisar o trade-off entre a eficiência matemática na interceptação de fraudes e o custo social gerado pelo bloqueio indevido de clientes legítimos (Falsos Positivos), propondo o uso de Aprendizado Sensível ao Custo (Cost-Sensitive Learning) para mitigar o viés algorítmico contra perfis de consumo voláteis.

Base de Dados
Os algoritmos foram treinados e validados utilizando a base de dados realista "IEEE-CIS Fraud Detection", disponibilizada publicamente na plataforma Kaggle. Devido ao tamanho do dataset original, o arquivo CSV não está hospedado neste repositório. Para executar os testes, é necessário baixar a base "train_transaction.csv" no Kaggle e inseri-la na raiz do projeto.

Descrição dos Códigos
O repositório é composto por duas frentes de experimentação:

1. Experimento Principal de Machine Learning
Este script executa a avaliação comparativa entre três classificadores (XGBoost, Random Forest e Regressão Logística). O código testa o desempenho de cada modelo em quatro cenários distintos de tratamento para o desbalanceamento extremo de classes:

Baseline (Sem tratamento de dados)

Random Under-Sampling

SMOTE (Synthetic Minority Over-sampling Technique)

Class Weight (Penalização na Função de Custo)

O código gera automaticamente as métricas de avaliação (AUC-PR, ROC-AUC, F1-Score e Tempo de Treinamento), exportando tabelas e gráficos comparativos para embasar a seção de Resultados do TCC. A conclusão técnica aponta o XGBoost com Class Weight como a arquitetura mais equilibrada e viável para sistemas em tempo real.

2. Análise de Custo Social (rede_neural.py)
Este script foi desenvolvido para testar a hipótese da hiper-rigidez algorítmica. Ele implementa uma Rede Neural Artificial profunda (Multilayer Perceptron) com três camadas ocultas (128, 64, 32) e aplica um limiar de decisão restritivo (20%).

O objetivo deste código não é a precisão absoluta, mas sim a extração de métricas de impacto social. Ele gera:

Um relatório de texto (.txt) quantificando o número exato de clientes inocentes bloqueados (Falsos Positivos) na tentativa de capturar os fraudadores.

Uma Matriz de Confusão para visualização dos acertos e erros.

Um gráfico de distribuição de probabilidades provando o conflito matemático entre as transações de clientes com comportamento volátil e os perfis fraudulentos.

Tecnologias Utilizadas
Linguagem: Python 3

Manipulação de Dados: Pandas, NumPy

Machine Learning: Scikit-learn

Visualização: Matplotlib, Seaborn

Como Executar
Clone o repositório em sua máquina local.

Certifique-se de ter o Python instalado e crie um ambiente virtual.

Instale as dependências executando: pip install pandas numpy scikit-learn matplotlib seaborn

Insira o arquivo "train_transaction.csv" na mesma pasta dos scripts.

Execute os códigos via terminal (ex: python rede_neural.py) para gerar os relatórios e gráficos localmente.
