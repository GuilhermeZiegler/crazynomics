# crazynomics
Este aplicativo filtra dados do mercado financeiro para os seguintes tipos de ativos:
moedas, commodites, indíces e ativos 
Carregamento dos dados:
  > Via série histórica yahoo finance
  > Via excel com index de data em pd.datetime()
Filtros disponíveis:
  > Data 
  > Open, High, Low, Close, Adj Close, Volume
Criação de variáveis:
  > Amplitude (open - close)
  > Amplitude máxima (high - low)
  > Amplitude e Amplitude máximas percentuais
  > Updown (if open > close: 1, else: 0)
  > Resultado Diário (close(t) - close(t-1)
  > Retorno diário  (close(t) - close(t-1) / close(t-1)
Filtros de processamento:
  > Remoção de coluna por estrutura Open_, Low_, Close_, High_, Volume, Ticker_, Adj Close_  e demais variáveis criadas (remover colunas)
  > Remoçãod e colunas mantendo apenas as desejadas (manter colunas)
Processamento de dados:
  > Média Móvel em NaN por coluna (não aplica uma média móvel na coluan inteira como o método rolling)
  > Seleção de agregação por frequência (diária, semanal, quinzenal, mensal, bimestral, trimestral, quadrimestral, semestral, anual, diária)
  > Agregação da frequência por média, mediana, soma e valor exato (para o caso de agregar do diário para mensal mantendo valores exatos de referência da data mês)
Visualização Gráfica:
  > Geração de gráficos de valor por tempo (sobrepostos)
  > Geração de gráficos candlesticke agregados (necessidade de High, Low, Open, Close selecionadas no filtro)
Fuções de séries temporais:
  > Identifica qual necessidade de diferenciação para cada variável que torna a série estacionária (make_stationary)
  > Identifica se vaiáveis estão estacionárias para uma janela de tempo por variável com abordagem constante, para frente e para trás (Stationary Window)
  > Identifica cointegração nas variáveis que compoem a série temporal tomando a variável y como referência com abordagem constante, para frente e para trás (Testar Cointegração)
  > Identifica variáveis cointegradas por análise combinatória definidas temporalmente (Implementação futura)
Modelos Temporais:
  > Função autoarima: com configuração de  Max_p, Max_q, d, test, seasonal, m e proporção de split train test
  > Função SARIMALL: otimizador de SARIMA por métricas de predição rmse, rme, mae, mape e critérios aic e bic. Configuração combinatória dos parâmetros p,q,d,P,Q,D,lags de acordo com limites estabelecidos.
  > Função SARIMALL: fit no modelo exato quando parâmetros p,q,d,P,Q,D diferentes de -1 e lags = fixo
  > Função AutoVAR: fit modelos combinatórios de VAR com as variáveis selecionadas para a diferenciação considerada. Opção de filtro por teste de causalidade de granger (restrição de variáveis) e métrica de erro  rmse, rme, mae, mape
   . Possibilidade de trabalhar com modelo na diferença ou realizar o ajuste da métrica após a integração up wind e nivelação dos dados.
>  AutoVAR estrutural: analisa todas as combinações possíveis na matriz decomposição de Cholesky e retorna a melhor composição para métricas de previsão
>  AutoVEC: (implementar)
