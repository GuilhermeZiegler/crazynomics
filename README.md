# crazynomics

Este aplicativo filtra dados do mercado financeiro para os seguintes tipos de ativos: moedas, commodities, índices e ativos.
Acesso: https://crazynomics.streamlit.app/

## Carregamento dos dados

- Via série histórica Yahoo Finance.
- Via arquivo Excel com índice de data em `pd.datetime()`.

## Filtros disponíveis

Você pode aplicar os seguintes filtros aos dados:

- Data
- Abertura (Open)
- Máxima (High)
- Mínima (Low)
- Fechamento (Close)
- Fechamento Ajustado (Adj Close)
- Volume

## Criação de variáveis

O aplicativo oferece a capacidade de criar várias variáveis a partir dos dados, incluindo:

- Amplitude (Open - Close)
- Amplitude Máxima (High - Low)
- Amplitude e Amplitude Máxima em percentagem
- Updown (1 se Open > Close, senão 0)
- Resultado Diário (Close(t) - Close(t-1))
- Retorno Diário (Close(t) - Close(t-1)) / Close(t-1)

## Filtros de processamento

Você pode realizar o seguinte processamento nos dados:

- Remoção de colunas por estrutura, como Open_, Low_, Close_, High_, Volume, Ticker_, Adj Close_, e outras variáveis criadas (remover colunas).
- Manter apenas as colunas desejadas (manter colunas).

## Processamento de dados

- Média Móvel em NaN por coluna (não aplica uma média móvel na coluna inteira como o método rolling).
- Seleção de agregação por frequência (diária, semanal, quinzenal, mensal, bimestral, trimestral, quadrimestral, semestral, anual).
- Agregação da frequência por média, mediana, soma e valor exato (para o caso de agregar do diário para mensal mantendo valores exatos de referência da data do mês).

## Visualização Gráfica

- Geração de gráficos de valor ao longo do tempo (sobrepostos).
- Geração de gráficos candlestick agregados (necessário selecionar High, Low, Open, Close no filtro).

## Funções de séries temporais

- Identificação da necessidade de diferenciação para cada variável para tornar a série estacionária (make_stationary).
- Identificação de variáveis estacionárias para uma janela de tempo por variável com abordagens constante, para frente e para trás (Stationary Window).
- Identificação de cointegração nas variáveis que compõem a série temporal, tomando a variável y como referência com abordagens constante, para frente e para trás (Testar Cointegração).
- Identificação de variáveis cointegradas por análise combinatória definida temporalmente (Implementação futura).

## Modelos Temporais

- Função autoarima: com configuração de Max_p, Max_q, d, test, seasonal, m e proporção de split train test.
- Função SARIMALL: otimizador de SARIMA por métricas de predição RMSE, RME, MAE, MAPE e critérios AIC e BIC. Configuração combinatória dos parâmetros p, q, d, P, Q, D, lags de acordo com limites estabelecidos.
- Função SARIMALL: ajuste no modelo exato quando os parâmetros p, q, d, P, Q, D são diferentes de -1 e lags são fixos.
- Função AutoVAR: ajuste de modelos combinatórios de VAR com as variáveis selecionadas para a diferenciação considerada. Opção de filtro por teste de causalidade de Granger (restrição de variáveis) e métrica de erro RMSE, RME, MAE, MAPE. Possibilidade de trabalhar com modelo na diferença ou realizar o ajuste da métrica após a integração upwind e nivelamento dos dados.
- AutoVAR estrutural: analisa todas as combinações possíveis na matriz de decomposição de Cholesky e retorna a melhor composição para métricas de previsão.
- AutoVEC: (implementação futura)

