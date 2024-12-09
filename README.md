# Aplicação_de_Machine_Learning_em_Classificação_para_o_Planejamento_da_Demanda_Probabilística_de_Recursos_Críticos_Submarinos

# Projeto Final
Aluna: Carolina Meduna Baziewicz

Orientadora: Prof. Dra. Manoela Kohler

Trabalho apresentado ao curso BI MASTER como pré-requisito para conclusão de curso pela PUC-Rio.

------------------------------------------------------------------------------------------------------------------------------
**Resumo**
Neste trabalho, foram aplicados dois algoritmos de classificação de *machine learning* num conjunto de dados real: o *suport vector machine* (SVM) e o *random forest* (ID3). A base de dados consiste em 4 colunas de entrada para 1 coluna de saída com aproximadamente 120.000 linhas, em que todos os dados são categóricos e a coluna de classificação possui duas opções de resposta: "Determinístico" ou "Probabilístico". Os métodos demonstraram excelentes resultados de acurácia, ambos superiores a 99%. Porém o tempo de treinamento do algoritmo Random Forest foi significativamente menor, tornando-se a melhor na comparação. A aplicação se demonstra adequada e bastante eficiente para o uso ao qual se propõe.

**Abstract**
In this work, two machine learning classification algorithms were applied to a real dataset: the *support vector machine* (SVM) and the *random forest* (ID3). The database consists of 4 input columns to 1 output column with approximately 120,000 rows, in which all data are categorical and the classification column has two options: "Deterministic" or "Probabilistic". The methods demonstrated excellent accuracy results, both above 99%. However, the training time of the Random Forest algorithm was significantly shorter, making it the best in the comparison. The application proves to be adequate and quite efficient for the use for which it is proposed.

# 1. Introdução e Objetivos
A aplicação de algoritmos de machine learning para problemas de classificação é uma abordagem poderosa e flexível para resolver uma ampla gama de desafios. A chave para o sucesso está em preparar bem os dados, escolher o algoritmo certo, treinar adequadamente o modelo e avaliar seu desempenho de forma rigorosa. Com o avanço das técnicas de ML, algoritmos cada vez mais sofisticados têm sido aplicados com sucesso em problemas complexos, oferecendo soluções inteligentes para problemas do cotidiano.

Para fins de aplicação e construção deste trabalho, foi escolhido como objeto de estudo um problema real do cotidiano de uma empresa do setor de óleo e gás cuja solução poderia se dar através dos algoritmos de *machine learning*.

Nesta empresa, as áreas clientes periodicamente são requisitadas a realizarem o levantamento da demanda necessária de serviços e materiais para o cumprimento das atividades previstas nos cronogramas dos projetos nos anos seguintes, tais como instalações de novas unidades, perfuração de novos poços, instalação de equipamentos, manutenção e substituição de linhas, integridade dos dutos e umbilicais submarinos instalados, abandono de poços, entre outros. Este processo é fundamental para identificação da necessidade de disponibilização das embarcações submarinas, conhecidas como recursos críticos (embarcações do tipo PLSV, AHTS, RSV, SDSV e SESV), bem como para sinalizar para a companhia o quanto ela deve se preparar física e financeiramente para cumprir com as atividades previstas nos projetos.

Parte da demanda informada neste processo é diretamente identificada, pois possui escopo e prazo plenamente definidos e por isso denomina-se “demanda determinística”, enquanto outra parte desta demanda caracteriza-se por uma demanda de difícil previsibilidade, não possuem escopo e/ou prazo plenamente definidos, mas ainda com elevada frequência de ocorrência, denominada “demanda probabilística”. Não se sabe ao certo quando ou onde esta demanda irá ocorrer, porém devido à sua representatividade, faz-se necessário o planejamento de recursos para esta finalidade. 

Para realizar o referido planejamento, as áreas utilizam uma ferramenta de previsão com uso de estatística e inteligência artificial para auxiliar as áreas no levantamento da demanda probabilística, que utiliza como input principal o histórico de realização dos serviços. Para fazer uso desta ferramenta, as áreas clientes precisam necessariamente possuir de forma antecipada a série temporal histórica de ocorrência das ordens de serviço probabilísticas, e para isso necessitam de uma base classificada de ordens de serviço. No processo atual, a classificação da base ocorre de forma manual, sendo processada individualmente por um analista. Contudo, o processo de classificação manual além de trazer riscos para as informações, também onera a realização das atividades pelas gerências de planejamento.

Toda ordem de serviço pode ser classificada como "Determinística" ou "Probabilística". Quando determinado serviço é realizado com uma embarcação, a ordem de serviço associada fica registrada no sistema próprio da empresa. Em um mês, são geradas em média X novas ordens de serviço, fazendo com que a quantidade de registros acumulados ao longo de um período seja enorme.

Pensando neste problema, o objetivo deste trabalho consiste em validar o uso de machine learning para auxiliar na classificação da base de ordens de serviço, facilitando o uso da ferramenta recentemente desenvolvida, e tornando assim mais eficiente o processo de planejamento da empresa.

# 2. Desenvolvimento
A aplicação de algoritmos de machine learning (ML) para problemas de classificação é um processo estruturado que envolve várias etapas: a preparação dos dados, a escolha do algoritmo, o treinamento do modelo e a avaliação de seu desempenho.

## 2.1. Preparação dos Dados
O primeiro passo para aplicar um algoritmo de ML a um problema de classificação é preparar os dados. Isso envolve coletar e organizar um conjunto de dados que contenha exemplos rotulados — ou seja, exemplos de entradas (características ou variáveis) junto com suas respectivas classes ou rótulos. Esses dados são frequentemente divididos em dois conjuntos principais:

  Conjunto de treinamento: Usado para treinar o modelo de ML, ou seja, para ensinar o algoritmo a fazer previsões.
  Conjunto de teste: Usado para avaliar o desempenho do modelo treinado em dados que ele ainda não viu, simulando o comportamento do modelo em novos dados.

Além disso, é importante realizar algumas etapas de pré-processamento dos dados, como a normalização ou padronização das variáveis, tratamento de valores ausentes e conversão de variáveis categóricas em numéricas.

## 2.2 Escolha do Algoritmo
Existem diversos algoritmos de classificação em machine learning, e a escolha do melhor depende do tipo de dados, da complexidade do problema e dos requisitos de desempenho. Neste trabalho, foram utilizados dois algortimos diferentes, para fins de comparação de uso e desempenho de acurácia: o Suport Vector Machine (SVM) e o Árvore de Decisão ID3.

## 2.2.1 Suport Vector Machine
O Support Vector Machine (SVM) é um dos métodos de aprendizado supervisionado mais poderosos e amplamente utilizados para problemas de classificação. Ele busca encontrar um hiperplano que melhor separa os dados de diferentes classes em um espaço de alta dimensão. Sua popularidade se deve à eficácia em classificar dados não lineares e ao conceito sólido de margem de separação.

Em um problema de classificação binária, onde as classes são representadas por dois grupos distintos de pontos de dados, o SVM tenta encontrar o hiperplano que separa essas classes de forma que a margem entre os dois grupos seja maximizada. A margem é definida como a distância entre o hiperplano e os pontos de dados mais próximos de cada classe, chamados de vetores de suporte. Esses vetores de suporte são essenciais para a definição do modelo e para a construção do hiperplano, pois qualquer ponto de dados fora deles não afeta diretamente a posição do hiperplano. A ideia por trás da maximização da margem é que um modelo com uma margem maior tem um melhor desempenho de generalização, ou seja, é mais provável que ele se comporte bem com novos dados que não foram usados durante o treinamento.

O hiperplano é uma linha (ou plano) que divide o espaço de características de modo que os dados de uma classe fiquem de um lado e os dados da outra classe fiquem do outro. Em um problema de duas dimensões, esse hiperplano é uma linha reta, enquanto em problemas de mais de duas dimensões, o hiperplano se torna um plano ou um espaço de maior dimensão. Se os dados forem linearmente separáveis, a tarefa do SVM é encontrar o hiperplano que maximiza a margem. No entanto, muitos problemas de classificação envolvem dados que não são linearmente separáveis. Nesses casos, o SVM pode ser estendido para lidar com dados não lineares.

O que torna o SVM tão poderoso em problemas não lineares é o uso da técnica de kernel. O kernel permite que o SVM trabalhe em um espaço de características de dimensão mais alta, onde os dados podem ser separáveis linearmente, mesmo que não sejam separáveis no espaço original. Em vez de transformar explicitamente os dados para um espaço de maior dimensão, o kernel realiza esse mapeamento de forma implícita, economizando tempo computacional. Existem vários tipos de funções kernel, como o kernel linear, polinomial, RBF (Radial Basis Function) e outros. O kernel RBF é frequentemente usado em casos práticos devido à sua capacidade de lidar com relações complexas entre as classes.

Em muitos cenários do mundo real, os dados não são perfeitamente separáveis. Para lidar com isso, o SVM introduz uma penalização chamada de margem suave. Em vez de insistir em uma separação perfeita, o SVM permite algumas violações da margem, ou seja, alguns pontos de dados podem ser classificados incorretamente, mas com um custo associado. Esse parâmetro de penalização é controlado pelo parâmetro C, que equilibra a busca pela margem máxima e a penalização por erros de classificação. Quando C é pequeno, o modelo permite mais erros (maior margem de separação, mas mais flexível), enquanto um C grande tenta minimizar os erros de classificação ao custo de uma margem menor.

Vantagens do SVM:
  - Alta performance: O SVM é eficaz mesmo em conjuntos de dados com alta dimensionalidade, o que o torna uma excelente escolha para problemas de classificação com muitas variáveis.
  - Boa generalização: Devido à maximização da margem, o SVM tende a ser bom em generalizar para dados novos, evitando o overfitting.
  - Flexibilidade com kernels: O uso de diferentes funções kernel permite que o SVM seja aplicado a uma ampla variedade de problemas, incluindo aqueles com fronteiras de decisão não lineares.

Desvantagens do SVM:
  - Custo computacional: O treinamento de um modelo SVM pode ser computacionalmente intensivo, especialmente em grandes conjuntos de dados. Isso ocorre porque o tempo de treinamento pode crescer rapidamente com o número de amostras e características.
  - Escolha do kernel: A escolha do kernel correto e a definição dos parâmetros adequados (como C e o parâmetro do kernel) podem ser desafiadoras e requerem ajuste fino.

## 2.2.2 Árvore de Decisão ID3
O algoritmo Árvore de Decisão ID3 (Iterative Dichotomiser 3) é um dos métodos mais tradicionais e amplamente utilizados em machine learning para problemas de classificação. Ele constrói uma árvore de decisão de forma recursiva, onde cada nó interno representa uma decisão com base em um atributo dos dados, e cada folha da árvore corresponde a uma classe ou categoria. A construção da árvore visa dividir o espaço de características de maneira a maximizar a pureza das classes em cada nó.

O algoritmo ID3 é baseado na ideia de dividir os dados em subconjuntos com o máximo de homogeneidade possível em relação à classe alvo. Para isso, o ID3 utiliza um critério de seleção de atributos chamado ganho de informação. O ganho de informação mede a redução da incerteza (ou entropia) sobre a classe alvo ao dividir o conjunto de dados com base em um atributo específico.

Vantagens do ID3:
  - Simplicidade e Interpretabilidade: O ID3 gera árvores fáceis de entender e interpretar, o que facilita a explicação dos resultados para especialistas não técnicos.
  - Eficiência: O algoritmo é relativamente eficiente em termos de tempo de execução, especialmente em conjuntos de dados menores e mais simples.
  - Aplicabilidade: Funciona bem em problemas de classificação onde as variáveis independentes são discretas ou podem ser discretizadas.

Desvantagens do ID3:
  - Sobreajuste: O algoritmo pode criar árvores muito grandes e complexas, especialmente se o conjunto de dados for muito ruidoso ou contiver muitos atributos irrelevantes. Isso leva ao sobreajuste, em que o modelo se adapta muito aos dados de treinamento e tem um desempenho ruim em novos dados.
  - Dependência de atributos discretos: O ID3 é mais adequado para atributos discretos e pode exigir transformações nos dados para lidar com atributos contínuos.
  - Ganho de Informação e Atributos com Muitos Valores: A escolha do atributo a ser dividido pode ser enviesada para atributos com muitos valores, uma vez que eles podem produzir divisões mais puras, mas que não são necessariamente as mais informativas.

## 2.3 Treinamento do Modelo
Após a escolha do algoritmo, o próximo passo é treinar o modelo. O treinamento envolve ajustar os parâmetros do modelo com base no conjunto de dados de treinamento. Durante o treinamento, o algoritmo tenta aprender padrões a partir das características dos dados que permitem prever corretamente as classes. A técnica de otimização usada varia de acordo com o algoritmo, mas o objetivo final é minimizar o erro de previsão, ou seja, tornar o modelo capaz de classificar corretamente os exemplos do conjunto de treinamento.

## 2.4 Avaliação do Modelo
Uma vez treinado, o modelo é avaliado usando o conjunto de teste. A avaliação consiste em medir o desempenho do modelo em dados que não foram usados durante o treinamento. Para problemas de classificação, algumas métricas comuns de avaliação incluem:

  - Acurácia: Proporção de previsões corretas em relação ao total de previsões feitas.
  - Precisão: Proporção de exemplos classificados como positivos que realmente são positivos.
  - Recall (sensibilidade): Proporção de exemplos positivos corretamente identificados pelo modelo.
  - F1-Score: A média harmônica entre precisão e recall, útil quando há um desequilíbrio nas classes.
  
Além disso, dependendo do problema, pode ser necessário ajustar o modelo para melhorar seu desempenho, por meio de técnicas como validação cruzada, ajuste de hiperparâmetros (tuning) e regularização.

# 3. Resultados
Nesta seção serão apresentados os resultados obtidos pela aplicação dos Algoritmos SVM e Árvore de Decisão ID3 no conjunto de dados.

## 3.1 Descrição do Conjunto de Dados e Tratamento
Conforme comentado anteriormente, para este estudo foi utilizada uma base real de dados da empresa de registro das ordens de serviço realizadas por embarcações submarinas de uma empresa no ramo de óleo e gás. A quantidade de colunas original foi reduzida através da escolha das colunas principais, bem como o tratamento de *missings* efetuado. A base tratada é composta por 5 colunas e aproximadamente 120.000 linhas. Para realização da etapa de treinamento e testes, foi utilizada 90% da base original, e os 10% restantes foi utilizado para verificação dos modelos. Todas as colunas da base são compostas por atributos categóricos. A coluna classificação possui duas opções de resposta: "Determinístico" ou "Probabilístico". Para efeitos de Ilustração, na Tabela 1 abaixo foi apresentada uma amostra do conjunto de Dados.

| Classe de Serviço | Classe de Entrega | Classe de Demanda | Origem | Classificação |
|------------------------|------------------------|------------------------|------------------------|------------------------|
| Inspeção Programada PIDF-2 | Manutenção e Operação de Sistemas Submarinos | Inspeção de Dutos | Espírito Santo | Determinístico |
| Instalação de Acessórios | Emergência/Contingência | Dados em Sistema Submarinos e/ou de Ancoragem | Bacia de Campos | Probabilístico |
| Lançamento de Âncora | Implantação de Novos Sistemas Submarinos | Ancoragem de Unidade Marítima | Bacia de Santos | Determinístico |
| Manuseio de Válvula (MDV) | Manutenção e Operação de Sistemas Submarinos e de Ancoragem | Operação de Equipamentos Submarinos | Espírito Santos | Probabilístico |

        Tabela 1 - Amostra do Conjunto de Dados

A Tabela a seguir apresenta a quantidade de atributos distintos por coluna, na base estudada.

| Classe de Serviço | Classe de Entrega | Classe de Demanda | Origem | Classificação |
|------------------------|------------------------|------------------------|------------------------|------------------------|
| 556 | 4 | 54 | 12| 2 |

Tabela 2 - Quantidade de Atributos Distintos por Coluna

## 3.2 Aplicação do Algoritmo Suport Vector Machine (SVM)

A seguir, os resultados obtidos na Etapa de Treinamento:
  - Duração Etapa de Treinamento (conjunto treino): 22m20.6s
  - Duração Etapa de Previsão e Avaliação (conjunto teste): 50seg
  - Acurácia: 0.9990622069396686
  - Kappa: 0.9961762355098488
  - F1: 0.9967234335180616
  - Quantidade de Acertos: 86.292
  - Quantidade de Erros: 81

![image](https://github.com/user-attachments/assets/e9e87918-32a9-4495-9b01-843dc153c7a7)

        Figura 1 - Matriz de Confusão - Resultado de Treino - SVM

Resultados de Teste:
  - Acurácia Teste: 0.9986107252014449
  - Kappa Teste: 0.9943426409357636
  - F1 Teste: 0.9951534733441034
  - Quantidade de Acertos: 21.564
  - Quantidade de Erros: 30

![image](https://github.com/user-attachments/assets/5ba51eba-85b7-4260-8521-5979df71397c)

        Figura 2 - Matriz de Confusão - Resultado de Teste - SVM

Resultados da Validação:
  - Quantidade de Dados: 11.997
  - Quantidade de Acertos: 
  - Quantidade de Erros: 
  - Acurácia Validação: 100%

## 3.3 Aplicação do Algoritmo Árvore de Decisão (ID3)
Neste capítulo serão apresentados os resultados da aplicação do algoritmo de Árvore de Decisão ID3 no conjunto de dados proposto.

![image](https://github.com/user-attachments/assets/bdff4fce-20a4-4e84-90a4-34a4525d6095)

        Figura 3 - Representação da Árvore de Decisão Gerada

Resultados de Treino:
  - Profundidade da Árvore: 14 níveis
  - Duração Etapa de Treinamento (conjunto treino): 1,3 seg
  - Duração Etapa de Previsão e Avaliação (conjunto treino): 0,2 seg
  - Acurácia Treino: 0.9990043184791544
  - Kappa Treino: 0.9959406115564705
  - F1 Treino: 0.9965215984468533
  - Quantidade de Acertos: 86.287
  - Quantidade de Erros: 86

![image](https://github.com/user-attachments/assets/9144eaae-7fb5-40bd-b252-9f2e32c4fdfa)

        Figura 4 - Matriz de Confusão - Resultados do Treino - Método ID3

Resultados de Teste:
  - Acurácia Teste: 0.9987033435213485
  - Kappa Teste: 0.994714098573155
  - F1 Teste: 0.9954707214493691
  - Quantidade de Acertos: 21.566
  - Quantidade de Erros: 28

![image](https://github.com/user-attachments/assets/f8b795ff-b4e9-4411-8e63-a95ae1f5afca)

        Figura 5 - Matriz de Confusão - Resultados do Teste - Método ID3

Resultados da Validação:
  - Quantidade de Dados: 11.997
  - Quantidade de Acertos: 11.997
  - Quantidade de Erros: 0
  - Acurácia Validação: 100%


## 3.4 Análise Comparativa
A análise dos resultados do capítulo anterior nos permite traçar uma análise comparativa entre os métodos utilizados.
De forma geral, ambos os métodos se demonstraram excelentes para o problema em questão. A acurácia de ambos é bastante similar, e o percentual de Acurácia tanto no conjunto de treino, quanto teste e validação é superior a 99%.
Entretanto, o tempo computacional requerido para treinar o modelo SVM é significativamente maior. Enquanto o SVM executou a etapa de treinamento em 22minutos, o algoritmo Árvore de Decisão demorou apenas 1,3 seg.

| Método | Tempo Treinamento Modelo | Qtde Acertos Treino | Qtde Erros Treino | Acurácia |
|------------------------|------------------------|------------------------|------------------------| ------------------------|
| SVM | 22m20seg | 86.292 | 81 | 0.99906 |
| Árvore de Decisão | 1,3 seg | 86.287 | 86 | 0.99900 | 

| Método | Tempo Treinamento Modelo | Qtde Acertos Teste | Qtde Erros Teste | Acurácia |
|------------------------|------------------------|------------------------|------------------------| ------------------------|
| SVM | 50seg | 21.564 | 30 | 0.99861 |
| Árvore de Decisão | 0,2seg | 21.566 | 28 | 0.99870 | 

| Método | Tempo Treinamento Modelo | Qtde Acertos Validação | Qtde Erros Validação | Acurácia |
|------------------------|------------------------|------------------------|------------------------| ------------------------|
| SVM |  |  |  |  |
| Árvore de Decisão |  |  |  |  | 

        Tabela 2 - Comparação dos Resultados Treino, Teste e Validação

Desta forma, pode-se concluir que para este conjunto de dados, o algoritmo de melhor performance foi a Árvore de Decisão ID3.

# 4. Conclusões
Neste trabalho foram aplicados dois métodos diferentes de machine learning para um problema clássico de classificação, em um contexto real do cotidiano de uma empresa.

A primeira técnica aplicado (SVM) se demonstrou bastante eficaz, apesar do elevado tempo de treinamento do modelo. O SVM é uma técnica robusta e eficaz para classificação, especialmente quando se trata de dados de alta dimensão e problemas de fronteiras de decisão complexas. Seu princípio fundamental de maximizar a margem de separação entre as classes, combinado com o uso de kernels para trabalhar com dados não lineares, torna-o uma ferramenta poderosa em diversos domínios, como reconhecimento de padrões, bioinformática e aprendizado de máquinas em geral.

A segunda técnica aplicada (Árvore de Decisão ID3) se demonstrou superior neste caso, pela elevada acurácia e baixíssimo tempo de resposta. O algoritmo Árvore de Decisão ID3 é uma técnica poderosa e intuitiva para classificação, baseada na ideia de dividir recursivamente os dados em subconjuntos mais homogêneos em relação às classes. Embora seja simples e eficaz para problemas pequenos e moderados, ele pode sofrer com problemas de sobreajuste e pode não ser ideal para todos os tipos de dados. No entanto, sua explicabilidade e a facilidade com que pode ser implementado tornam o ID3 uma ferramenta valiosa, especialmente em cenários em que a interpretabilidade do modelo é crucial.

Para a aplicação em questão, será de muita utilidade trazer para uso na empresa uma ferramenta robusta como esta, reduzindo significativamente o tempo de trabalho das equipes de planejamento, que atualmente realizam esta etapa de forma manual.


---------------------------------------------------------
Pós Graduação em Business Inteligence Master

Universidade Pontificia Católica do Rio de Janeiro
