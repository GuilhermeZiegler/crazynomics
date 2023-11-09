##  Importação do streamlit para exibição da Dashboard
import streamlit as st

## Importação de bibliotecas de manipulação padrão
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta, datetime, time
import openpyxl
from openpyxl import Workbook
from math import sqrt

## importação de bibliotecas de plotagem de visualziação de dados
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go


## importação de bibliotecas de strings
import os
import io
import requests
import warnings
import copy
import pyarrow.parquet as pq
import json

import unidecode
import re
import string
import joblib
import glob


# Tratamentos gerais 
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Modelos de séries temporais
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from itertools import product, combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pmdarima as pm
from statsmodels.tsa.api import VAR



# demais modelos
import optuna
from category_encoders import JamesSteinEncoder, WOEEncoder, CatBoostEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# importação de dados
import yfinance as yf

def get_cached_data():
    return {}
cached_data = get_cached_data()
dfs = pd.DataFrame()
## Funções para a monografia


# Acha a primeira linha inteira que não contem NA
def firstrow_notna(df):
    for index, row in df.iterrows():
        if row.notna().all():
            return index
    return None

# Acha a ultima linha inteira que não contem NA
def lastrow_notna(df):
    last_complete_index = None
    for index, row in df.iterrows():
        if row.notna().all():
            last_complete_index = index
    return last_complete_index

MAX_MESSAGES = 50  # Limite máximo de mensagens no histórico
HISTORY_FILE = "chat_history.json"  # Nome do arquivo de histórico
        
# Função para carregar o histórico do arquivo
def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    else:
        return []
    
# Função para salvar o histórico no arquivo
def save_chat_history(messages):
    with open(HISTORY_FILE, "w") as file:
        json.dump(messages, file)

# Função para adicionar mensagens ao histórico
def chat(message, sender, messages):
    messages.append({"role": sender, "content": message})
    
@st.cache_data
def filtra_dados(df, merged_df,start_date,end_date):
    filtered_columns = []

    for prefix in merged_df.index.to_list():
        columns_to_keep = [col for col in df.columns if col.endswith(prefix)]
        filtered_columns.extend(columns_to_keep)

    dfs = df[filtered_columns]
    first_index = firstrow_notna(dfs)
    last_index = lastrow_notna(dfs)
    first_ind = pd.to_datetime(first_index)
    last_ind = pd.to_datetime(last_index)
    
    if first_index is not None and pd.to_datetime(first_ind) > pd.to_datetime(start_date):
        st.warning(f'Data de início ótima: {first_index}', icon="⚠️")
    
    if last_index is not None and pd.to_datetime(last_ind) < pd.to_datetime(end_date):
        st.warning(f'Data de término ótima: {last_index}', icon="⚠️")
    
    dfs.drop(dfs[(dfs.index < first_ind) | (dfs.index > last_ind)].index, inplace=True)
    
    dfs.drop(dfs[(dfs.index < pd.to_datetime(start_date)) | (dfs.index > pd.to_datetime(end_date))].index, inplace=True)
    return dfs
       
@st.cache_data
def heavycleaning(dfs, limpeza_pesada):
    for prefix in limpeza_pesada:
        columns_to_drop = [col for col in dfs.columns if col.startswith(prefix)]
        dfs.drop(columns=columns_to_drop, inplace=True)
    st.write('Quantidade de NaN:', dfs.isna().sum().to_frame().T)
    st.write("Tamanho após a limpeza Pesada: ", dfs.shape)
    return dfs 

@st.cache_data
def guardar_coluna(dfs, colunas_keep):
    columns_to_keep = [col for col in dfs.columns if col in colunas_keep]
    dfs = dfs[columns_to_keep]
    dfs.columns = columns_to_keep  # Atualiza as colunas no DataFrame original
    st.write('Quantidade de NaN:', dfs.isna().sum().to_frame().T)
    st.write("Tamanho após o processamento ", dfs.shape)
    return dfs


def cortar_volume(dfs, corte_volume):
    columns_to_drop = [col for col in dfs.columns if col.startswith('Volume_') and (dfs[col] == 0).sum() >= corte_volume]
    dfs.drop(columns=columns_to_drop, inplace=True)
    st.write('Quantidade de NaN:', dfs.isna().sum().to_frame().T)
    st.write("Tamanho após corte de % Volumes Zerados na coluna:", dfs.shape)
    return dfs
   
def fill_moving_avg(dfs, window_size):
    date_index = dfs.index
    dfs.reset_index(drop=True, inplace=True)
    for col in dfs.select_dtypes(include=[np.number]).columns:  # Seleciona apenas colunas numéricas
        nan_indices = dfs[dfs[col].isna()].index  # Obtém os índices com valores NaN na coluna
        for index in nan_indices:
            start = max(0, index - window_size)  # Início da janela deslizante
            end = index + 1  # Fim da janela deslizante
            window_data = dfs[col].iloc[start:end]  # Seleciona os valores dentro da janela
            mean_value = round(window_data.mean(), 3)  # Calcula a média com 3 casas decimais
            dfs.at[index, col] = mean_value 

    dfs.index = date_index  # Restaura o índice original fora do loop

    st.write("Tamanho após aplicar médias móveis:", dfs.shape)
    st.write(f'Média Móvel em dias: {window_size}')
    st.write(f'Quantidade de NaN: {dfs.isna().sum().sum()} dados nulos')
    st.write(dfs.isna().sum().to_frame().T)

def frequencia_amostral(df, frequencia_desejada, funcao_agregacao='sum'):
    frequencia_para_periodo = {
        'diária': 'D',
        'semanal': 'W',
        'quinzenal': '2W',
        'mensal': 'M',
        'bimestral': '2M',
        'trimestral': '3M',
        'quadrimestral': '4M',
        'semestral': '6M',
        'anual': 'A'
    }

    if frequencia_desejada not in frequencia_para_periodo:
        raise ValueError("Frequência desejada não suportada.")

    periodo = frequencia_para_periodo[frequencia_desejada]
    
    resampled_series = df.resample(periodo)
    
    if funcao_agregacao == 'sum':
        result_series = resampled_series.sum()
    elif funcao_agregacao == 'mean':
        result_series = resampled_series.mean()
    elif funcao_agregacao == 'median':
        result_series = resampled_series.median()
    elif funcao_agregacao == 'valor_exato':
        result_series = df[df.index.isin(resampled_series.first().index)]
    else:
        raise ValueError("Função de agregação não suportada.")
    
    return result_series

def read_parquet_file():
    response = requests.get(link)
    buffer = io.BytesIO(response.content)
    df = pd.read_parquet(buffer)
    st.write('Tamanho total: ', df.shape)
    return df

def read_excel_file(excel):
    df = pd.read_excel(excel)
    return df

def criar_variaveis(dfs, variaveis_selecionadas):
    sufixos_unicos = set()
    for coluna in dfs.columns:
        partes = coluna.split('_')
        if len(partes) > 1:
            sufixos_unicos.add(partes[-1])

    # Crie um dicionário de mapeamento de variáveis
    mapeamento_variaveis = {
        'Resultado_diario': ('Close', 'Open'),
        'Amplitude': ('High', 'Low'),
        'Retorno_diario': ('Close',),
        'Maxima_variacao_amplitude': ('Amplitude',),
        'Variacao_amplitude_diaria': ('Resultado_diario',),
        'Updown': ('Retorno_diario',)
    }

    for sufixo in sufixos_unicos:
        for variavel, colunas_necessarias in mapeamento_variaveis.items():
            if variavel in variaveis_selecionadas:  # Verifique se a variável está na lista de selecionadas
                if all(coluna + "_" + sufixo in dfs.columns for coluna in colunas_necessarias):
                    if variavel == 'Resultado_diario':
                        dfs[f'Resultado_diario_{sufixo}'] = dfs[f'Close_{sufixo}'] - dfs[f'Open_{sufixo}']
                    elif variavel == 'Amplitude':
                        dfs[f'Amplitude_{sufixo}'] = dfs[f'High_{sufixo}'] - dfs[f'Low_{sufixo}']
                    elif variavel == 'Retorno_diario':
                        dfs[f'Retorno_diario_{sufixo}'] = dfs[f'Close_{sufixo}'].pct_change().fillna(0)
                    elif variavel == 'Maxima_variacao_amplitude':
                        dfs[f'Maxima_variacao_amplitude_{sufixo}'] = dfs[f'Amplitude_{sufixo}'].pct_change().fillna(0)
                    elif variavel == 'Normal_Amplitude_Change':
                        dfs[f'Variacao_amplitude_diaria_{sufixo}'] = dfs[f'Resultado_diario_{sufixo}'].pct_change().fillna(0)
                    elif variavel == 'Updown':
                        dfs[f'Updown_{sufixo}'] = (dfs[f'Retorno_diario_{sufixo}'] > 0).astype(int)						
    return dfs

def get_candle(dfs, list_of_dictionaries):
    processed_columns = set()

    for col in dfs.columns:
        if '_' in col:
            suffix = col.split('_')[-1]
        else:
            suffix = col

        for dictionary in list_of_dictionaries:
            # Verificar se o argumento é iterável
            if hasattr(dictionary, '__iter__'):
                if suffix in dictionary:
                    processed_columns.add(suffix)
                    break
            else:
                # Lida com o caso em que o argumento não é iterável
                if suffix == dictionary:
                    processed_columns.add(suffix)
                    break
			
    return list(processed_columns)

def candlestick_chart(dfs, selected_suffixes):
    traces = []
    for suffix in selected_suffixes:
        open_col = f"Open_{suffix}"
        high_col = f"High_{suffix}"
        low_col = f"Low_{suffix}"
        close_col = f"Close_{suffix}"

        trace = go.Candlestick(
            x=dfs.index,
            open=dfs[open_col],
            high=dfs[high_col],
            low=dfs[low_col],
            close=dfs[close_col],
            name=suffix
        )

        traces.append(trace)

    layout = go.Layout(
        title="Gráfico de Candlestick",
        xaxis=dict(title="Data"),
        yaxis=dict(title="Preço"),
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig


def make_stationary(df, N_MAX):
    VARVEC_diff = df.copy()
    df_resultante = pd.DataFrame(index=VARVEC_diff.columns)
    estacionarias_set = set()

    for i in range(N_MAX):
        if i == 0:
            pass
        else:
            VARVEC_diff = VARVEC_diff[nao_estacionarias].diff()
            VARVEC_diff.dropna(inplace=True)

        _, estacionarias, nao_estacionarias = adfuller_trintinalia(VARVEC_diff)
        estacionarias_set.update(estacionarias)

        for col in VARVEC_diff.columns:
            if col in estacionarias_set:
                if f'estacionarias_{i}' not in df_resultante.columns:
                    df_resultante[f'estacionarias_{i}'] = ''
                if df_resultante.loc[col, f'estacionarias_{i}'] != 'S':
                    df_resultante.loc[col, f'estacionarias_{i}'] = 'S'
            
        if len(nao_estacionarias) == 0:
            st.write(f"Todas as variáveis se tornaram estacionárias após {i} diferenciações.")
            break
        elif i == N_MAX - 1:
            st.write(f"Atingido o número máximo de iterações ({N_MAX}) e algumas variáveis ainda não são estacionárias.")
            
    st.write(df_resultante)

def adfuller_trintinalia(VARVEC, nivel_critico=5):
    result_df = pd.DataFrame(columns=['Variável', 'ADF Statistic', 'p-value', 'Nc_1%', 'Nc_5%', 'Nc_10%', 'ResNc_1%', 'ResNc_5%', 'ResNc_10%'])
    estacionarias = []
    nao_estacionarias = []

    for col in VARVEC.columns:
        result = adfuller(VARVEC[col].values)
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        res_nc = adf_statistic < critical_values[f'{nivel_critico}%']

        row = {
            'Variável': col,
            'ADF Statistic': adf_statistic,
            'p-value': p_value,
            'Nc_1%': critical_values['1%'],
            'Nc_5%': critical_values['5%'],
            'Nc_10%': critical_values['10%'],
            'ResNc_1%': 'ESTACIONARIO' if adf_statistic < critical_values['1%'] else 'NAO ESTACIONARIO',
            'ResNc_5%': 'ESTACIONARIO' if res_nc else 'NAO ESTACIONARIO',
            'ResNc_10%': 'ESTACIONARIO' if adf_statistic < critical_values['10%'] else 'NAO ESTACIONARIO'
        }

        result_df = pd.concat([result_df, pd.DataFrame(row, index=[0])], ignore_index=True)

        if res_nc:
            estacionarias.append(col)
        else:
            nao_estacionarias.append(col)

    if nivel_critico == 1:
        nao_estacionarias = estacionarias.copy()
    
    
    result_df = pd.DataFrame(result_df)
    estacionarias_df = pd.DataFrame({'Variáveis Estacionárias': estacionarias})
    nao_estacionarias_df = pd.DataFrame({'Variáveis Não Estacionárias': nao_estacionarias})
    
    return result_df, estacionarias, nao_estacionarias

def stationary_window_adf_multi_columns(dataframe, window_size, approach, offset_type='M'):
    results = {}
    max_index = dataframe.index.max()
    min_index = dataframe.index.min()
    offset_mapping = {
        'D': 'days',
        'BD': 'weekday',
        'W': 'weeks',
        'M': 'months',
        'Y': 'years',
        'MIN': 'minutes',
        'H': 'hours',
        'S': 'seconds',
        'MS': 'microseconds',
        'MS_START': 'months=window_size, day=1',
        'MS_END': 'months=window_size, day=1'
    }
    
    if offset_type not in offset_mapping:
        supported_options = ', '.join(offset_mapping.keys())
        raise ValueError(f"The offset_type parameter must be one of the supported options: {supported_options}")
    else:
        if offset_type in ['MS_START', 'MS_END']:
            offset_value = offset_mapping[offset_type].replace('window_size', str(window_size))
            offset = pd.DateOffset(eval(offset_value))
        else:
            offset = pd.DateOffset(**{offset_mapping[offset_type]: window_size})
    
    for column in dataframe.columns:
        df = dataframe[column]
        data_inicio = min_index
        data_fim = data_inicio + offset
        column_results = []
        
        if approach == 'constant':
            while data_fim <= max_index:
                df_slice = df.loc[data_inicio:data_fim]
                result = adfuller(df_slice)
                p_value = result[1]
                is_stationary = p_value <= 0.05
                column_results.append((data_inicio, data_fim, p_value, is_stationary, "constant"))
                data_inicio = data_fim
                data_fim += offset
        elif approach == 'forward':
            while data_fim <= max_index:
                df_slice = df.loc[data_inicio:data_fim]
                result = adfuller(df_slice)
                p_value = result[1]
                is_stationary = p_value <= 0.05
                column_results.append((data_inicio, data_fim, p_value, is_stationary, approach))
                data_fim += offset
        elif approach == 'back':
            data_fim = max_index
            data_inicio = data_fim - offset
            while data_inicio >= min_index:
                df_slice = df.loc[data_inicio:data_fim]
                result = adfuller(df_slice)
                p_value = result[1]
                is_stationary = p_value <= 0.05
                column_results.append((data_inicio, data_fim, p_value, is_stationary, approach))
                data_inicio -= offset

        results[column] = pd.DataFrame(column_results, columns=['Start Date', 'End Date', 'p-value', 'Is Stationary', 'Approach'])

        
    return results

def johansen_cointegration_test(df, variavel_y, variaveis_coint, det_order=-1, k_ar_diff=0, nc="5%"):
    if nc == "1%":
        col = 0
    elif nc == "5%":
        col = 1
    elif nc == "10%":
        col = 2
    else:
        raise ValueError("NC = 1%, 10% OU 5%")

    resultado = {}
    data_inicio = df.index.min()
    data_fim = df.index.max()
    variavel_interesse = df[variavel_y]
    df_outras_variaveis = df[variaveis_coint]
    variaveis_cointegradas = ', '.join(variaveis_coint)
    series_list = {"Variável de Interesse": variavel_interesse}

    for col_name in variaveis_coint:
        series_list[col_name] = df_outras_variaveis[col_name]

    data = pd.DataFrame(series_list)
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    autovalores = result.eig
    trace_statistics = result.lr1
    eigen_value_statistics = result.lr2
    critical_values_trace = result.cvt
    critical_maximum_eigenvalue_statistic = result.cvm
    ranking_trace = 0
    cointegra_trace = 0
    cointegra_eigenvalue = 0
    ranking_eigenvalue = 0
    size = len(trace_statistics)

    for i in range(0, size):
        if trace_statistics[i] >= critical_values_trace[i][col]:
            cointegra_trace = 1
            ranking_trace += 1
        if eigen_value_statistics[i] >= critical_maximum_eigenvalue_statistic[i][col]:
            cointegra_eigenvalue = 1
            ranking_eigenvalue += 1

    resultado = {
        "Data_Inicio": data_inicio,
        "Data_Fim": data_fim,
        "Variável de Interesse": variavel_interesse.name,
        "Variáveis Coint": variaveis_cointegradas,
        "Cointegração (Trace)": cointegra_trace,
        "Ranking (Trace)": ranking_trace,
        "Cointegração (Max Eigenvalue)": cointegra_eigenvalue,
        "Ranking (Max Eigenvalue)": ranking_eigenvalue,
        "Autovalores": autovalores  # Inclui os autovalores nos resultados
    }

    resultado_df = pd.DataFrame([resultado])  # Crie um DataFrame a partir do dicionário resultado
    return resultado_df

def coint_window(df, offset_type, window_size, approach, variavel_y, variaveis_coint,det_order=-1, k_ar_diff=0, nc="5%"):
    result = []
    max_index = df.index.max()
    min_index = df.index.min()
    offset_mapping = {
        'D': 'days',
        'BD': 'weekday',
        'W': 'weeks',
        'M': 'months',
        'Y': 'years',
        'MIN': 'minutes',
        'H': 'hours',
        'S': 'seconds',
        'MS': 'microseconds',
        'MS_START': 'months=window_size, day=1',
        'MS_END': 'months=window_size, day=1'
    }
    
    if offset_type not in offset_mapping:
        supported_options = ', '.join(offset_mapping.keys())
        raise ValueError(f"The offset_type parameter must be one of the supported options: {supported_options}")
    else:
        if offset_type in ['MS_START', 'MS_END']:
            offset_value = offset_mapping[offset_type].replace('window_size', str(window_size))
            offset = pd.DateOffset(eval(offset_value))
        else:
            offset = pd.DateOffset(**{offset_mapping[offset_type]: window_size})
    
    data_inicio = min_index
    data_fim = data_inicio + offset

    if approach == 'constant':
        while data_fim <= max_index:
            df_slice = df.loc[data_inicio:data_fim]   
            slice_result = johansen_cointegration_test(df_slice, variavel_y, variaveis_coint,det_order, k_ar_diff)
            result.append(slice_result)
            data_inicio = data_fim
            data_fim += offset
    elif approach == 'forward':
        while data_fim <= max_index:
            df_slice = df.loc[data_inicio:data_fim]
            slice_result = johansen_cointegration_test(df_slice, variavel_y, variaveis_coint, det_order, k_ar_diff)
            result.append(slice_result)
            data_fim += offset
    elif approach == 'back':
        data_fim = max_index
        data_inicio = data_fim - offset
        while data_inicio >= min_index:
            df_slice = df.loc[data_inicio:data_fim]
            slice_result = johansen_cointegration_test(df_slice, variavel_y, variaveis_coint,det_order, k_ar_diff)
            result.append(slice_result)
            data_inicio -= offset
			
    combined_result = pd.concat(result, ignore_index=True)
    
    return combined_result

@st.cache_data
def my_auto_arima(cut, df,variavel,teste,d, max_p,max_q,seasonal,m):
	filterd_df = df[variavel]
	if filterd_df.isna().any().any():
		st.warning("Dados NaN encontrados. Por favor, processe o dataframe")
		return
	
	train_size = int(len(filterd_df) * (1 - cut))
	train_df = filterd_df.iloc[:train_size]
	test_df = filterd_df.iloc[train_size:]
	st.write('Tamanho total do df: ', len(df))
	st.write('Tamanho de treino df:', len(train_df))
	st.write('Tamanho de teste df:', len(test_df))				  
	model = pm.auto_arima(train_df, 
						  test=teste, 
						  start_p=1, 
						  d=d, 
						  start_q=1,
                          max_p=max_p, 
						  max_q=max_q,
                          seasonal=seasonal,
						  m=m,
                          stepwise=True, trace=True,
                          error_action='ignore',
                          suppress_warnings=True)

	predictions = model.predict(n_periods=len(test_df))
	forecast_df = pd.DataFrame()
	forecast_df['forecast_OOT'] = predictions
	forecast_df.index = test_df.index
	df = df[[variavel]]
	df = df.merge(forecast_df, left_index=True, right_index=True, how ="outer")
	with st.container():
		st.write(model.summary())
	fig = px.line(df, x = df.index, y=df.columns,
						 title ="AutoArima") 
	with st.container():
		st.plotly_chart(fig)

def SARIMALL(cut, df, variavel, stationarity, p, d, q, P, D, Q, limite_combinacoes, lags, metric,variar_lag, n_plots):
	filtered_df = df[variavel]
	if filtered_df.isna().any().any():
		st.warning("Dados NaN encontrados. Por favor, processe o dataframe")
		return

	train_size = int(len(df) * (1 - cut))
	train_df = filtered_df.iloc[:train_size]
	test_df = filtered_df.iloc[train_size:]
	st.write('Tamanho total do df: ', len(filtered_df))
	st.write('Tamanho de treino df:', len(train_df))
	st.write('Tamanho de teste df:', len(test_df))
	resultados_list = []
	params = {'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q}  # Defina os parâmetros como um dicionário

	if params is not None:
		for key, value in params.items():
			if value == -1:
				params[key] = None

	p, d, D, P, Q, q = params.values()

	param_ranges = [list(range(limite_combinacoes + 1)) if param is None else [param] for param in [p, d, q, P, D, Q]]
	if variar_lag == "Fixo":
		lags_values = [lags]  # Se lags deve ser fixo, use o valor especificado
	else:
		lags_values = list(range(1, lags + 1)) 
	param_ranges.append(lags_values)
	param_combinations = list(product(*param_ranges))
	resultados_df = pd.DataFrame(columns=["p", "d", "q", "P", "D", "Q", "lags", "aic","bic","rmse","mse","mae","mape"])
	st.write("Total de modelos para o otimizador: ", len(param_combinations))
	for params in param_combinations:
		p, d, q, P, D, Q, lags = params
		try:
			modelo = SARIMAX(train_df, 
							 order=(p, d, q),
							 seasonal_order=(P, D, Q, lags),
							 enforce_stationarity=stationarity)
			resultado_modelo = modelo.fit()
			y_pred = resultado_modelo.get_forecast(steps=len(test_df)).predicted_mean
			rmse = sqrt(mean_squared_error(test_df, y_pred))
			mae = mean_absolute_error(test_df, y_pred)  # Exemplo de outra métrica
			mse = mean_squared_error(test_df, y_pred)
			mape = mean_absolute_percentage_error(test_df, y_pred)
			aic = resultado_modelo.aic
			bic = resultado_modelo.bic
			resultados_list.append({'p': p, 'd': d, 'q': q, 
									'P': P, 'D': D, 'Q': Q,
									'lags': lags, 'aic': aic, 'bic': bic, 
									'rmse': rmse, 'mse': mse, 'mae': mae, 
									'mape': mape})
		except Exception as e:
			continue
	
	resultados_df = pd.DataFrame(resultados_list)             
	resultados_df = resultados_df.sort_values(by=metric)
	df_previsoes = pd.DataFrame(index=test_df.index)
	if n_plots <= len(resultados_df): 
		melhores_configuracoes = resultados_df.head(n_plots)
		for _, row in melhores_configuracoes.iterrows():
			p, d, q, P, D, Q, lags = row[['p', 'd', 'q', 'P', 'D', 'Q', 'lags']]
			modelo = SARIMAX(train_df, 
							 order=(p, d, q),
							 seasonal_order=(P, D, Q, lags),
							 enforce_stationarity=stationarity)
			resultado_modelo = modelo.fit()
			y_pred = resultado_modelo.get_forecast(steps=len(test_df)).predicted_mean
			y_pred.index = test_df.index
			coluna = f'Previsão (p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, lags={lags})'
			df_previsoes[coluna] = y_pred
			
		df_completo = pd.DataFrame(filtered_df).merge(df_previsoes, left_index=True, right_index=True, how='outer')
		fig = px.line(df_completo, x = df_completo.index, y=df_completo.columns,
						  title ="SARIMALL")
		st.plotly_chart(fig)
	else: 
		st.write("Reduza n_plots para convergir com a quantidade de modelos otimizados")
	st.markdown("Resultados Otimizados")
	st.dataframe(resultados_df)
	return resultados_df

def grangercausalitytests_trintinalia(df, y_column, max_lags, n, nc=0.05, x_column="ALL", VAR_SELECT=False):
    # Aplicar a diferenciação e remoção de NaNs fora do loop
    for _ in range(n):
        df = df.diff()
        df.dropna(inplace=True)

    p_values_df = pd.DataFrame(columns=['Y', 'X', 'Lag', 'P-valor', 'Granger Causa'])
    granger_causa_dict = {}

    if x_column == "ALL":
        x_columns = [col_name for col_name in df.columns if col_name != y_column]
    else:
        x_columns = x_column

    for col_name_X in x_columns:
        cg_df = df[[y_column, col_name_X]]
        result = grangercausalitytests(cg_df, max_lags, verbose=False)
        p_vals = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lags + 1)]
        temp_df = pd.DataFrame({
            'Y': y_column,
            'X': col_name_X,
            'Lag': list(range(1, max_lags + 1)),
            'P-valor': p_vals,
            'Granger Causa': [f"{y_column} é granger causado por {col_name_X}" if p < nc
                             else f"{y_column} não é granger causado por {col_name_X}" for p in p_vals]
        })

        p_values_df = pd.concat([p_values_df, temp_df], ignore_index=True)
    
    df_filtrado = p_values_df[p_values_df['P-valor'] < nc]

    df_lags = pd.DataFrame(columns=['X', 'Lags'])
    variaveis_x = df_filtrado['X'].unique()

    for x in variaveis_x:
        df_x = df_filtrado[df_filtrado['X'] == x]
        lags = ', '.join([f'{x}' for x in df_x['Lag'].astype(str)])
        df_lags.loc[len(df_lags)] = [x, lags]

    df_lags = df_lags.rename(columns={'X': f'Granger Causam {y_column}'})
    g1, g2 = st.columns(2)
    with g1:
        st.dataframe(df_lags)
    with g2:
        st.dataframe(p_values_df)
    
    dfs = pd.DataFrame()
    if VAR_SELECT:
        colunas_filtradas = [y_column] + list(df_lags[f'Granger Causam {y_column}'])
        session_state.data = session_state.data[colunas_filtradas]

    return session_state.data 

def AUTOVAR(df, vardiff, cut, max_lags, var_y, x_columns):
    variables = [var_y] + x_columns 
    df_combo = []
    for _ in range(vardiff):
        df = df.diff()
        df.dropna(inplace=True)
    for r in range(1, len(variables) + 1):
        combos = combinations(variables, r)
        for combo in combos:
            if combo[0] == var_y and len(combo) > 1:
                df_combo.append(combo)  # Remova list() aqui
    st.write(f"Foram gerados {len(df_combo)} conjuntos para o modelo VAR")
    model_colnames_list = []
    lags_list = []
    rmse_list = []
    posicao_combinacao_list = []
    predicted_dfs = []
    train_dfs = []
    train_size  = int(len(df) * (1 - cut))
    
    for posicao, df_combinacao in enumerate(df_combo):
        if all(col in df.columns for col in df_combinacao):
            VARn_combinacao = df[list(df_combinacao)]
            
            rmse_dict = {}
            
            for k in range(1, max_lags + 1):
                train_df = VARn_combinacao.iloc[:train_size, :]
                test_df = VARn_combinacao.iloc[train_size:, :]
                model = VAR(train_df)
                fitted_model = model.fit(maxlags=k)
                
                n_forecast = len(test_df)
                forecast = fitted_model.forecast(y=test_df.values, steps=n_forecast)
                
                predicted_df = pd.DataFrame(forecast, columns=test_df.columns, index=test_df.index)
                rmse = sqrt(mean_squared_error(test_df[var_y], predicted_df[var_y]))
                
                rmse_dict[k] = rmse
        
            lag_otimo = min(rmse_dict, key=rmse_dict.get)
            modelo_name = f"Modelo_lag{lag_otimo}"
            concatenated_colnames = " ".join(train_df.columns)
            
            model_colnames_list.append(concatenated_colnames)
            lags_list.append(lag_otimo)
            rmse_list.append(rmse_dict[lag_otimo])
            posicao_combinacao_list.append(posicao)
            train_dfs.append(train_df)
            predicted_dfs.append(predicted_df)
    
    df_resultante = pd.DataFrame({
        'model_colnames': model_colnames_list,
        'Lags': lags_list,
        'RMSE': rmse_list,
        'posicao_da_combinacao': posicao_combinacao_list})
    df_resultante = df_resultante.sort_values(by="RMSE")
    st.dataframe(df_resultante)
    
# Plotar todas as previsões no mesmo gráfico com Plotly Express
    fig = px.line()
    fig.add_scatter(x=train_df.index, y=train_df[var_y], mode='lines', name='Treinamento')
    fig.add_scatter(x=test_df.index, y=test_df[var_y], mode='lines', name='Teste')

    for idx, row in df_resultante.head(10).iterrows():  # Plotar as previsões dos 10 melhores modelos
        lag = row['Lags']
        predicted_df = predicted_dfs[row['posicao_da_combinacao']]
        fig.add_scatter(x=predicted_df.index, y=predicted_df[var_y], mode='lines', name=f'Previsão (Lags={lag})')

    fig.update_layout(
        title="Previsões dos 10 Melhores Modelos",
        xaxis_title="Data",
        yaxis_title=var_y,
        showlegend=True,
    )

    st.plotly_chart(fig)

def gerar_betas(df, colunas):
    beta_df = pd.DataFrame(index=colunas, columns=colunas)

    # Loop para calcular os coeficientes beta
    for col1 in colunas:
        for col2 in colunas:
            if col1 != col2:
                X = np.log(df[col2] / df[col2].shift(1)).dropna()  # Variável independente
                y = np.log(df[col1] / df[col1].shift(1)).dropna()  # Variável dependente
                common_index = X.index.intersection(y.index)  # Índices comuns
                
                if len(common_index) > 0:  # Verificar se há dados comuns
                    X = X[common_index]
                    y = y[common_index]
                    X = sm.add_constant(X)  # Adicionar constante ao modelo
                    model = sm.OLS(y, X).fit()  # Ajustar o modelo de regressão linear
                    beta = model.params[1]  # Coeficiente beta (o índice 1 representa a variável independente)
                    beta_df.at[col1, col2] = beta
    beta_df.fillna(1, inplace = True)
     # Remoção da parte superior da diagonal
    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            beta_df.iloc[i, j] = None
    # Criação do heatmap triangular
    fig = px.imshow(
        beta_df,
        color_continuous_scale=[[0, 'red'], [0.5, 'white'], [1, 'blue']],
        zmin=-2,
        zmax=2,
        labels=dict(x="Ativo", y="Ativo", color="Coeficiente Beta"),
        x=colunas,
        y=colunas,
        title="Heatmap Dinâmico dos Coeficientes Beta",
        width=1000,
        height=800)
   
    st.plotly_chart(fig)	

## Configuração da página e do título
st.set_page_config(page_title='Monografia Guilherme Ziegler', layout = 'wide', initial_sidebar_state = 'auto')

st.title("Processador de séries temporais")

link = 'https://github.com/GuilhermeZiegler/crazynomics/raw/master/dados.parquet'

st.subheader('Download da base histórica', help='Você deve clicar no botão carregue a base para que o arquivo parquet seja lido',divider='rainbow')

# Importar SessionState
class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

@st.cache(allow_output_mutation=True)
def get_session():
    return SessionState(df=None, data=None, cleaned=False, messages=None)

session_state = get_session()

if st.button('Carregue a base'):
	session_state.data = read_parquet_file()
	st.write('Base histórica lida com sucesso!')
	
excel_file = st.file_uploader("Selecione um arquivo em excel para leitura", type=["xlsx", "xls"])
ler_excel = st.button("Carregar Excel")
if ler_excel and excel_file is not None:
	session_state.data = read_excel_file(excel_file)
	st.write("Excel carregado com sucesso")
else: 
	st.warning('É preciso subir um arquivo válido em "browse files"')
## Bloco de datas para slicer

max_date = dt.datetime.today().date()
min_date = (dt.datetime.today() - dt.timedelta(days=10 * 700)).date()

try:
    default_start = df.index.min().strftime('%Y-%m-%d')
    default_end = df.index.max().strftime('%Y-%m-%d')
except:
    default_start = '2004-07-01'
    default_end   = dt.datetime.today().date().strftime('%Y-%m-%d')

dt1, dt2 =st.columns(2)	
	
default_start = dt.datetime.strptime(default_start, '%Y-%m-%d').date()
default_end = dt.datetime.strptime(default_end, '%Y-%m-%d').date()

with dt1:
	selected_start = st.date_input("Data Início:", min_value=default_start, max_value=default_end, value=default_start)
with dt2:	
	selected_end = st.date_input("Data Fim:", min_value=default_start, max_value=default_end,value=default_end)
	
start_date = selected_start
end_date = selected_end  

st.write(f"Dados para o range de datas selecionado: {start_date} até {end_date}")

coins, comm, assets, idx   = st.columns(4)

with coins:
## Bloco de moedas 
	moedas = {
			'JPY': 'USDJPY=X',
			'BRL': 'BRL=X',
			'ARS': 'ARS=X',
			'PYG': 'PYG=X',
			'UYU': 'UYU=X',
			'CNY': 'CNY=X',
			'KRW': 'KRW=X',
			'MXN': 'MXN=X',
			'IDR': 'IDR=X',
			'EUR': 'EUR=X',
			'CAD': 'CAD=X',
			'GBP': 'GBP=X',
			'CHF': 'CHF=X',
			'AUD': 'AUD=X',
			'NZD': 'NZD=X',
			'HKD': 'HKD=X',
			'SGD': 'SGD=X',
			'INR': 'INR=X',
			'RUB': 'RUB=X',
			'ZAR': 'ZAR=X',
			'SEK': 'SEK=X',
			'NOK': 'NOK=X',
			'TRY': 'TRY=X',
			'AED': 'AED=X',
			'SAR': 'SAR=X',
			'THB': 'THB=X',
			'DKK': 'DKK=X',
			'MYR': 'MYR=X',
			'PLN': 'PLN=X',
			'EGP': 'EGP=X',
			'CZK': 'CZK=X',
			'ILS': 'ILS=X',
			'HUF': 'HUF=X',
			'PHP': 'PHP=X',
			'CLP': 'CLP=X',
			'COP': 'COP=X',
			'PEN': 'PEN=X',
			'KWD': 'KWD=X',
			'QAR': 'QAR=X',
			"Bitcoin USD": "BTC-USD",
			"Ethereum USD": "ETH-USD",
			"Tether USDt USD": "USDT-USD",
			"BNB USD": "BNB-USD",
			"XRP USD": "XRP-USD",
			"USD Coin USD": "USDC-USD",
			"Lido Staked ETH USD": "STETH-USD",
			"Cardano USD": "ADA-USD",
			"Dogecoin USD": "DOGE-USD",
			"Solana USD": "SOL-USD",
			"Wrapped TRON USD": "WTRX-USD",
			"TRON USD": "TRX-USD",
			"Wrapped Kava USD": "WKAVA-USD",
			"Polkadot USD": "DOT-USD",
			"Dai USD": "DAI-USD",
			"Polygon USD": "MATIC-USD",
			"Litecoin USD": "LTC-USD",
			"Shiba Inu USD": "SHIB-USD",
			"Toncoin USD": "TON11419-USD",
			"Wrapped Bitcoin USD": "WBTC-USD",
			"Bitcoin Cash USD": "BCH-USD",
			"UNUS SED LEO USD": "LEO-USD",
			"Avalanche USD": "AVAX-USD",
			"Stellar USD": "XLM-USD",
			"Chainlink USD": "LINK-USD",
			}

	moedas_mono = {
			'JPY': 'USDJPY=X',
			'BRL': 'BRL=X',
			'ARS': 'ARS=X',
			'PYG': 'PYG=X',
			'UYU': 'UYU=X',
			'CNY': 'CNY=X',
			'KRW': 'KRW=X',
			'MXN': 'MXN=X',
			'IDR': 'IDR=X',
			'EUR': 'EUR=X',
			'CAD': 'CAD=X'
			}
	st.info("Moedas")
	default_moedas = moedas_mono.keys()
	moedas_keys = list(moedas.keys())
	moedas_selecionadas = st.multiselect("", moedas_keys)
	if moedas_selecionadas:
		# Criar um DataFrame a partir do dicionário filtrando as moedas selecionadas
		df_moedas = pd.DataFrame(moedas.items(), columns=['Moeda', 'Ticker']).set_index('Moeda').loc[moedas_selecionadas]
		# Exibir o DataFrame resultante
		#st.write(df_moedas)
	else:
		df_moedas  = pd.DataFrame()
## Bloco de commodities

with comm:
	lista_commodities = {
						"Brent Crude Oil Last Day Financ": "BZ=F",
						"Cocoa": "CC=F",
						"Coffee": "KC=F",
						"Copper": "HG=F",
						"Corn Futures": "ZC=F",
						"Cotton": "CT=F",
						"Heating Oil": "HO=F",
						"KC HRW Wheat Futures": "KE=F",
						"Lean Hogs Futures": "HE=F",
						"Live Cattle Futures": "LE=F",
						"Mont Belvieu LDH Propane (OPIS)": "B0=F",
						"Natural Gas": "NG=F",
						"Orange Juice": "OJ=F",
						"OURO": "GC=F",
						"Oat Futures": "ZO=F",
						"Palladium": "PA=F",
						"PETROLEO CRU": "CL=F",
						"Platinum": "PL=F",
						"RBOB Gasoline": "RB=F",
						"Random Length Lumber Futures": "LBS=F",
						"Rough Rice Futures": "ZR=F",
						"Silver": "SI=F",
						"Soybean Futures": "ZS=F",
						"Soybean Oil Futures": "ZL=F",
						"S&P Composite 1500 ESG Tilted I": "ZM=F",
						"Sugar #11": "SB=F",
						"WisdomTree International High D": "GF=F"
						}

	commodities_mono  = {'Soybean Futures':'ZS=F',  # Soja
						'Soybean Oil Futures': 'ZL=F', # Óleo de soja 
						'Live Cattle Futures':'LE=F'  # Boi gordo
						   }

	default_commodities = commodities_mono.keys()
	st.info("Commodities")
	commodities_keys = list(lista_commodities.keys())
	commodities_selecionadas = st.multiselect("", commodities_keys)
	if commodities_selecionadas:
		# Criar um DataFrame a partir do dicionário filtrando as moedas selecionadas
		df_commodities = pd.DataFrame(lista_commodities.items(), columns=['Commodities', 'Ticker']).set_index('Commodities').loc[commodities_selecionadas]
			# Exibir o DataFrame resultante
			#st.write(df_commodities)
	else:
		df_commodities = pd.DataFrame()
	## Bloco de empresas
with assets: 
	empresas_mono = {
					'MARFRIG': 'MRFG3.SA',
					'BRF': 'BRFS3.SA',
					'MINERVA': 'BEEF3.SA',
					'JBS': 'JBSS3.SA',
					'AGRO3':'AGRO3.SA'
					}

	lista_empresas = {
		"3M": "MMMC34.SA",
		"Abbott Laboratories": "ABTT34.SA",
		"AES Brasil": "AESB3.SA",
		"AF Invest": "AFHI11.SA",
		"Afluente T": "AFLT3.SA",
		"Agribrasil": "GRAO3.SA",
		"AgroGalaxy": "AGXY3.SA",
		"Aliansce Sonae": "ALSO3.SA",
		"Alliar": "AALR3.SA",
		"Alper": "APER3.SA",
		"Alphabet": "GOGL35.SA",
		"Alupar Investimento": "ALUP4.SA",
		"Amc Entert H": "A2MC34.SA",
		"American Express": "AXPB34.SA",
		"Apple": "AAPL34.SA",
		"Arcelor": "ARMT34.SA",
		"Att Inc": "ATTB34.SA",
		"Att Inc": "ATTB34.SA",
		"Auren Energia": "AURE3.SA",
		"Avalara Inc": "A2VL34.SA",
		"Avon": "AVON34.SA",
		"Banco do Brasil": "BBAS11.SA",
		"Banco Inter": "BIDI3.SA",
		"Banco Mercantil de Investimentos": "BMIN3.SA",
		"Banco Pan": "BPAN4.SA",
		"Bank America": "BOAC34.SA",
		"Banpara": "BPAR3.SA",
		"Banrisul": "BRSR3.SA",
		"Battistella": "BTTL3.SA",
		"Baumer": "BALM3.SA",
		"BB Seguridade": "BBSE3.SA",
		"Beyond Meat": "B2YN34.SA",
		"Biomm": "BIOM3.SA",
		"Biotoscana": "GBIO33.SA",
		"BMG": "BMGB11.SA",
		"Brasil Brokers": "BBRK3.SA",
		"brMalls": "BRML3.SA",
		"BTG S&P 500 CI": "SPXB11.SA",
		"BTG SMLL CAPCI": "SMAB11.SA",
		"Caesars Entt": "C2ZR34.SA",
		"Caixa Agências": "CXAG11.SA",
		"Camden Prop": "C2PT34.SA",
		"CAMIL": "CAML3.SA",
		"Carrefour": "CRFB3.SA",
		"CARTESIA FIICI": "CACR11.SA",
		"Casan": "CASN4.SA",
		"CEB": "CEBR6.SA",
		"CEEE-D": "CEED4.SA",
		"Ceee-gt": "EEEL4.SA",
		"CEG": "CEGR3.SA",
		"Celesc": "CLSC4.SA",
		"Celpe": "CEPE6.SA",
		"Celulose Irani": "RANI3.SA",
		"Cemig": "CMIG4.SA",
		"CESP": "CESP6.SA",
		"Chevron": "CHVX34.SA",
		"Churchill Dw": "C2HD34.SA",
		"Cisco": "CSCO34.SA",
		"Citigroup": "CTGP34.SA",
		"ClearSale": "CLSA3.SA",
		"Coca-Cola": "COCA34.SA",
		"Coelce": "COCE6.SA",
		"Coinbase Glob": "C2OI34.SA",
		"Colgate": "COLG34.SA",
		"Comgás": "CGAS3.SA",
		"ConocoPhillips": "COPH34.SA",
		"COPEL UNT N2": "CPLE11.SA",
		"Copel": "CPLE6.SA",
		"CPFL Energia": "CPFE3.SA",
		"CSN": "CSNA3.SA",
		"CSU CardSyst": "CARD3.SA",
		"Cyrusone Inc": "C2ON34.SA",
		"Dexco": "DXCO3.SA",
		"Dexxos Part": "DEXP3.SA",
		"Dimed": "PNVL3.SA",
		"Dommo": "DMMO3.SA",
		"Doordash Inc": "D2AS34.SA",
		"Draftkings": "D2KN34.SA",
		"eBay": "EBAY34.SA",
		"Enauta Part": "ENAT3.SA",
		"Energisa MT": "ENMT3.SA",
		"Engie Brasil": "EGIE3.SA",
		"EQI RECECI": "EQIR11.SA",
		"Eucatex": "EUCA4.SA",
		"Exxon Mobil": "EXXO34.SA",
		"Ferbasa": "FESA4.SA",
		"FIAGRO JGP CI": "JGPX11.SA",
		"FIAGRO RIZA CI": "RZAG11.SA",
		"FII BRIO ME CI": "BIME11.SA",
		"FII CYRELA CI ES": "CYCR11.SA",
		"FII GTIS LG": "GTLG11.SA",
		"FII HUSI CI ES": "HUSI11.SA",
		"FII JS A FINCI": "JSAF11.SA",
		"FII MORE CRICI ER": "MORC11.SA",
		"FII PLUR URBCI": "PURB11.SA",
		"FII ROOFTOPICI": "ROOF11.SA",
		"Fleury": "FLRY3.SA",
		"Freeport": "FCXO34.SA",
		"FT CLOUD CPT": "BKYY39.SA",
		"FT DJ INTERN": "BFDN39.SA",
		"FT EQ OPPORT": "BFPX39.SA",
		"FT HCARE ALPH DRN": "BFXH39.SA",
		"FT INTL EQ OP": "BFPI39.SA",
		"FT MOR DV LEA": "BFDL39.SA",
		"FT NASD CYBER": "BCIR39.SA",
		"FT NASD100 EQ": "BQQW39.SA",
		"FT NASD100 TC": "BQTC39.SA",
		"FT NAT GAS": "BFCG39.SA",
		"FT NYSE BIOT DRN": "BFBI39.SA",
		"FT RISI DIVID": "BFDA39.SA",
		"FT TECH ALPH": "BFTA39.SA",
		"G2D Investments": "G2DI33.SA",
		"GE": "GEOO34.SA",
		"General Shopping": "GSHP3.SA",
		"Ger Paranapanema": "GEPA4.SA",
		"Gerdau": "GOAU4.SA",
		"Getnet": "GETT11.SA",
		"Godaddy Inc": "G2DD34.SA",
		"Goldman Sachs": "GSGI34.SA",
		"Gradiente": "IGBR3.SA",
		"Halliburton": "HALI34.SA",
		"Honeywell": "HONB34.SA",
		"HP Company": "HPQB34.SA",
		"Hypera Pharma": "HYPE3.SA",
		"IBM": "IBMB34.SA",
		"Iguatemi S.A.": "IGTI3.SA",
		"Infracommerce": "IFCM3.SA",
		"Instituto Hermes Pardini SA": "PARD3.SA",
		"Intel": "ITLC34.SA",
		"INVESTO ALUG": "ALUG11.SA",
		"INVESTO USTK": "USTK11.SA",
		"INVESTO WRLD": "WRLD11.SA",
		"IRB Brasil RE": "IRBR3.SA",
		"ISA CTEEP": "TRPL4.SA",
		"ISHARES CSMO": "CSMO.SA",
		"ISHARES MILA": "MILA.SA",
		"Itaú Unibanco": "ITUB4.SA",
		"Itaúsa": "ITSA4.SA",
		"JBS": "JBSS3.SA",
		"Johnson": "JNJB34.SA",
		"JPMorgan": "JPMC34.SA",
		"Kingsoft Chl": "K2CG34.SA",
		"Klabin S/A": "KLBN11.SA",
		"Linx": "LINX3.SA",
		"Livetech": "LVTC3.SA",
		"Locaweb": "LWSA3.SA",
		"Log": "LOGG3.SA",
		"LPS Brasil": "LPSB3.SA",
		"Marfrig": "MRFG3.SA",
		"Mastercard": "MSCD34.SA",
		"MDiasBranco": "MDIA3.SA",
		"Medical P Tr": "M2PW34.SA",
		"Mercantil do Brasil Financeira": "MERC4.SA",
		"Merck": "MRCK34.SA",
		"Microsoft": "MSFT34.SA",
		"Minerva": "BEEF3.SA",
		"MMX Mineração": "MMXM3.SA",
		"Morgan Stanley": "MSBR34.SA",
		"Msciglmivolf": "BCWV39.SA",
		"Multiplan": "MULT3.SA",
		"Natura": "NTCO3.SA",
		"Neoenergia": "NEOE3.SA",
		"Nu Holdings": "NUBR33.SA",
		"Nu Renda Ibov Smart Dividendos (NDIV11)": "NDIV11.SA",
		"OdontoPrev": "ODPV3.SA",
		"OI": "OIBR4.SA",
		"Omega Energia": "MEGA3.SA",
		"Oncoclínicas": "ONCO3.SA",
		"Oracle": "ORCL34.SA",
		"OSX Brasil": "OSXB3.SA",
		"Ourofino S/A": "OFSA3.SA",
		"Padtec": "PDTC3.SA",
		"Pão de Açúcar": "PCAR3.SA",
		"Paranapanema": "PMAM3.SA",
		"Pepsi": "PEPB34.SA",
		"Petrobras": "PETR4.SA",
		"PetroRecôncavo Geral SA": "RECV3.SA",
		"PETRORIO": "PRIO3.SA",
		"Pfizer": "PFIZ34.SA",
		"Porto Seguro": "PSSA3.SA",
		"PPLA": "PPLA11.SA",
		"Privalia": "PRVA3.SA",
		"Procter Gamble": "PGCO34.SA",
		"Proctor Gamble": "PGCO34.SA",
		"Qualcomm": "QCOM34.SA",
		"Qualicorp": "QUAL3.SA",
		"RD": "RADL3.SA",
		"Renova": "RNEW4.SA",
		"Rio Bravo": "RBIV11.SA",
		"Sabesp": "SBSP3.SA",
		"Sanepar": "SAPR4.SA",
		"Santander BR": "SANB11.SA",
		"Sao Carlos": "SCAR3.SA",
		"São Martinho": "SMTO3.SA",
		"Schlumberger": "SLBG34.SA",
		"Sea Ltd": "S2EA34.SA",
		"Shopify Inc": "S2HO34.SA",
		"Smart Fit": "SMFT3.SA",
		"Snowflake": "S2NW34.SA",
		"Sp500 Value": "BIVE39.SA",
		"Sp500growth": "BIVW39.SA",
		"Square Inc": "S2QU34.SA",
		"Squarespace": "S2QS34.SA",
		"Starbucks": "SBUB34.SA",
		"STONE CO": "STOC31.SA",
		"Store Capital": "S2TO34.SA",
		"Sun Commun": "S2UI34.SA",
		"Suzano Holding": "NEMO6.SA",
		"Suzano Papel": "SUZB3.SA",
		"SYN Prop Tech": "SYNE3.SA",
		"Taesa": "TAEE11.SA",
		"Teladochealt": "T2DH34.SA",
		"Telebras": "TELB4.SA",
		"Telefônica Brasil S.A": "VIVT3.SA",
		"Tellus Desenvolvimento Logístico": "TELD11.SA",
		"Terra Santa Agro SA": "LAND3.SA",
		"Tim Participações": "TIMS3.SA",
		"Totvs": "TOTS3.SA",
		"Trade Desk": "T2TD34.SA",
		"TradersClub": "TRAD3.SA",
		"Tronox": "CRPG6.SA",
		"Uipath Inc": "P2AT34.SA",
		"Ultrapar": "UGPA3.SA",
		"Unipar": "UNIP6.SA",
		"Unity Softwr": "U2ST34.SA",
		"Vale": "VALE5.SA",
		"Verizon": "VERZ34.SA",
		"Visa": "VISA34.SA",
		"Vivara": "VIVA3.SA",
		"Viveo": "VVEO3.SA",
		"Votorantim Asset Management": "VSEC11.SA",
		"Walmart": "WALM34.SA",
		"Wells Fargo": "WFCO34.SA",
		"West Pharma": "W2ST34.SA",
		"Wilson Sons": "PORT3.SA",
		"Xerox": "XRXB34.SA",
		"XP Inc": "XPBR31.SA",
		"Zynga Inc": "Z2NG34.SA",
		}


	st.info("Ativos")
	default_empresas = list(empresas_mono.keys())
	empresas_keys = list(lista_empresas.keys())
	empresas_selecionadas = st.multiselect("", empresas_keys)
	if empresas_selecionadas:
		# Criar um DataFrame a partir do dicionário filtrando as moedas selecionadas
		df_empresas = pd.DataFrame(lista_empresas.items(), columns=['Empresa', 'Ticker']).set_index('Empresa').loc[empresas_selecionadas]
			# Exibir o DataFrame resultante
			#st.write(df_empresas)
	else:
		df_empresas = pd.DataFrame()
## Bloco de commodities
indices_mono  = {'S&P GSCI':'GD=F', #  commodities agrícolas, incluindo soja, trigo, milho e algodão
                'IBOVESPA': '^BVSP',
               }
with idx:
	lista_indices  = {
					'S&P GSCI':'GD=F', #  commodities agrícolas, incluindo soja, trigo, milho e algodão
					'IBOVESPA': '^BVSP', # B4 
					'S&P/CLX IPSA': '^IPSA',
					'MERVAL': '^MERV',
					'IPC MEXICO': '^MXX',
					'S&P 500': '^GSPC',
					'Dow Jones Industrial Average': '^DJI',
					'NASDAQ Composite': '^IXIC',
					'NYSE COMPOSITE (DJ)': '^NYA',
					'NYSE AMEX COMPOSITE INDEX': '^XAX',
					'Russell 2000': '^RUT',
					'CBOE Volatility Index': '^VIX',
					'S&P/TSX Composite index': '^GSPTSE',
					'FTSE 100': '^FTSE',
					'DAX PERFORMANCE-INDEX': '^GDAXI',
					'CAC 40': '^FCHI',
					'ESTX 50 PR.EUR': '^STOXX50E',
					'Euronext 100 Index': '^N100',
					'BEL 20': '^BFX',
					'MOEX Russia Index': 'IMOEX.ME',
					'Nikkei 225': '^N225',
					'HANG SENG INDEX': '^HSI',
					'SSE Composite Index': '000001.SS',
					'Shenzhen Index': '399001.SZ',
					'STI Index': '^STI',
					'S&P/ASX 200': '^AXJO',
					'ALL ORDINARIES': '^AORD',
					'S&P BSE SENSEX': '^BSESN',
					'IDX COMPOSITE': '^JKSE',
					'FTSE Bursa Malaysia KLCI': '^KLSE',
					'S&P/NZX 50 INDEX GROSS ( GROSS': '^NZ50',
					'KOSPI Composite Index': '^KS11',
					'TSEC weighted index': '^TWII',
					'TA-125': '^TA125.TA',
					'Top 40 USD Net TRI Index': '^JN0U.JO',
					'NIFTY 50': '^NSEI'
					}

	st.info("Indices")            
	default_indices = list(indices_mono.keys())
	indices_keys = list(lista_indices.keys())
	indices_selecionados = st.multiselect("", indices_keys)
	if indices_selecionados:
		# Criar um DataFrame a partir do dicionário filtrando as moedas selecionadas
		df_indices = pd.DataFrame(lista_indices.items(), columns=['Indice', 'Ticker']).set_index('Indice').loc[indices_selecionados]
		# Exibir o DataFrame resultante
		#st.write(df_indices)
	else:
		df_indices = pd.DataFrame()
	## Funções para baixar os dados 

## Bloco de seleç]ao de variáveis
st.subheader("Variáveis selecionadas:", help = "Variáveis que você escolhe para filtrar em moedas, commodities, ativos e índices. Você pode remover, criar, modificar variáveis à esquerda", divider ="rainbow") 


merged_df = pd.concat([df_moedas, df_empresas, df_commodities, df_indices], axis=0)
exibe_df = merged_df.copy().index.to_list()
exibe_df = pd.DataFrame(exibe_df).T
st.dataframe(exibe_df)

start = start_date 
end = end_date
dfs = pd.DataFrame()

lista_variaveis= {
        'Resultado_diario': ('Close', 'Open'),
        'Amplitude': ('High', 'Low'),
        'Retorno_diario': ('Close',),
        'Maxima_variacao_amplitude': ('Amplitude',),
        'Variacao_amplitude_diaria': ('Resultado_diario',),
        'Updown': ('Retorno_diario',)
        }  
    

variaveis_keys = list(lista_variaveis.keys())
variaveis_selecionadas = st.sidebar.multiselect("Adicione Variáveis", variaveis_keys)

if st.sidebar.button("Criar Variáveis"):
    if session_state.data is not None:
        session_state.data = criar_variaveis(session_state.data, variaveis_selecionadas)

## bloco de limpeza pesada para tickers
prefixes = ['Open_', 'Close_','High_', 'Low_', 'Adj Close_', 'Volume_', 'Ticker_', 
            'Key_', 'Resultado_diario_', 'Amplitude_', 'Retorno_diario_', 
            'Maxima_variacao_amplitude_', 'Variacao_amplitude_diaria_','Updown_']


limpeza_pesada = st.sidebar.multiselect('Remova colunas TIPO_',prefixes)
if st.sidebar.button("Remover Colunas"):
	if session_state.data is not None:
		session_state.data = heavycleaning(session_state.data, limpeza_pesada)
	
colunas_keep = st.sidebar.multiselect('Selecione Colunas:', session_state.data.columns)
manter_colunas = st.sidebar.button("Manter Colunas")
if manter_colunas:
	if session_state.data is not None:
		session_state.data = guardar_coluna(session_state.data,colunas_keep)
b1, b2 = st.columns(2)
with b1:
	if st.button("Filtrar Dados"):
	    session_state.data = read_parquet_file()
	    session_state.data = filtra_dados(session_state.data, merged_df, start_date, end_date)
	else:
	    st.warning('Para carregar do excel: clique no botão carregar excel. É preciso trabalhar com um df indexado ou indexar a coluna de data!')
with b2:
	colunasdata =  ["None"] + list(session_state.data.columns)
	indexar_data = st.button("Aplicar indice de data")
	indice_data = st.selectbox("Selecione a coluna de data:", colunasdata)
	if indexar_data:
		session_state.data.set_index([indice_data], drop=True, inplace = True)

## bloco de corte por volume por percentual de zeros
#corte_volume = st.sidebar.slider('Remove Volume_ para percentual de 0 na coluna', 0, 100, 100, step=1)
#if st.sidebar.button("Cortar Volume"):
 #   if session_state.data is not None:
  #      cortar_volume(session_state.data, corte_volume)

dias_moving_avg =  st.sidebar.number_input('Dias para inputar média móvel:',1, 100, 3,step=1)                     

if st.sidebar.button("Média Móvel"):
    if session_state.data is not None:
            fill_moving_avg(session_state.data, dias_moving_avg)          

frequencia_para_periodo = {
        'diária': 'D',
        'semanal': 'W',
        'quinzenal': '2W',
        'mensal': 'M',
        'bimestral': '2M',
        'trimestral': '3M',
        'quadrimestral': '4M',
        'semestral': '6M',
        'anual': 'A'
    }			
frequencia_desejada = st.sidebar.selectbox("Selecione a frequência desejada:", list(frequencia_para_periodo.keys()))
funcao_agregacao = st.sidebar.selectbox("Selecione a função de agregação:", ['sum', 'mean', 'median', 'valor_exato'])
reamostrar = st.sidebar.button("Reamostrar Dados")
if reamostrar:
	if session_state is not None:
		session_state.data = frequencia_amostral(session_state.data,frequencia_desejada,funcao_agregacao)
			
# Exibição dos resultados
if session_state.data is not None:
	st.write("DataFrame:")
	st.dataframe(session_state.data)
	st.write(session_state.data.shape)

baixar_excel = st.button("Baixar Excel")
st.markdown('Pix para doações: guitziegler@gmail.com')
if baixar_excel:
    if session_state.data is not None:
        dfs_rounded = session_state.data.round(6)
        
        # Crie uma cópia do DataFrame para manter o índice original
        dfs_copy = dfs_rounded.copy()
        
        excel_file = "dados.xlsx"  # Nome do arquivo Excel a ser criado
        with st.spinner("Criando arquivo Excel..."):
            # Use o ExcelWriter do Pandas para criar o arquivo Excel
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                dfs_copy.to_excel(writer, index=True, header=True)
        st.success(f"Arquivo Excel criado com sucesso! Clique no botão abaixo para baixar.")
        st.download_button(
            label="Baixar dados em Excel",
            data=open(excel_file, "rb").read(),
            file_name=excel_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.subheader('Visualização Gráfica', help="Selecione um filtro de dados e clique na visualização gráfica disponível", divider='rainbow')

g1, g2 = st.columns(2)
	     	     
with g1:
	grafico_linhas = st.button("Gráfico Linhas")
	selected_columns = st.multiselect("Gráfico:", session_state.data.columns)
if grafico_linhas:
	if session_state.data is not None:
		if not selected_columns:
			st.warning("Selecione alguma coluna")
		else:
				# Crie o gráfico usando Plotly Express
			fig = px.line(session_state.data, x=session_state.data.index, y=selected_columns)
			with st.container():
					# Exiba o gráfico dentro do container
				st.plotly_chart(fig)
with g2:
	gerar_candles = st.button("Gerar Candles")
 	if gerar_candles:
		candles_tickers = get_candle(session_state.data, [lista_indices, lista_empresas, moedas, lista_commodities])
		selected_suffixes = st.multiselect("Selecione os sufixos:", candles_tickers)
	
	if st.button("Candlestick") and selected_suffixes:
	candlechart =  candlestick_chart(session_state.data, selected_suffixes)
	st.plotly_chart(candlechart)	
			
st.subheader('Modelos de Séries Temporais', help="Você poderá verificar estacionariedade das variáveis escolhidas e aplicar SARIMA, VAR e VEC", divider='rainbow')

p1, p2 = st.columns(2)
with p1:
	max_lags =  st.number_input('Lags:',1, 20, 3,step=1)   
	when_stationary = st.button("Make Stationary")

	if when_stationary:
		if session_state.data is not None:
			try:
				make_stationary(session_state.data, max_lags)
			except Exception as e:
				st.warning(f"Erro ao processar os dados: {str(e)}. Processo seus dados, remova NaN")
		else:
			st.warning("O dataframe não está disponível. Favor carregar dados primeiro.")

with p2:

	offset_mapping = {
			'D': 'days',
			'BD': 'weekday',
			'W': 'weeks',
			'M': 'months',
			'Y': 'years',
			'MIN': 'minutes',
			'H': 'hours',
			'S': 'seconds',
			'MS': 'microseconds',
			'MS_START': 'months=window_size, day=1',
			'MS_END': 'months=window_size, day=1'
		}

	freqs = list(offset_mapping.keys())
	freq = st.selectbox("Freq", freqs)
	window_size = st.number_input('Window Size:',1, 1000, 1,step=1) 
	approach =  st.radio("Selecione uma opção:", ["constant", "forward", "back"])
	stationary_window = st.button("Stationary window")
	if stationary_window:
		if session_state.data is not None:
			results = stationary_window_adf_multi_columns(session_state.data, window_size,approach,offset_type=freq)
			for column_name, result_df in results.items():
				st.write(result_df)

			
st.subheader('Verificador de  Coinregração', help="Testar Cointegração: Verifica cointegração por janela de tempo. Encontrar Coint: descobre quais combinações de ativos estão cointegradas para um intervalo de tempo e um tamanho ótimo. ATENÇÃO: Dados precisam estar com o index de data. Caso venha do filtro, automaticamente estarão neste formto.", divider='rainbow')	
	
co1, co2 = st.columns(2)	
with co1:
	nc = st.selectbox("Níveis críticos:",["5%","10%", "1%"])
	freqs_ = list(offset_mapping.keys())
	freq_ = st.selectbox("Freq Coint", freqs_)
	window_size_ = st.number_input('Coint Window Size:',1, 1000, 1,step=1) 
	approach_ =  st.radio("Coint opção:", ["constant", "forward", "back"])
	variavel_y = st.selectbox('Variável Y:', session_state.data.columns)
	variaveis_coint = st.multiselect('Variáveis Cointegrantes:',session_state.data.columns)
	det_order = st.number_input('-1 Auto, 0 None, 1 linear, 2 Square',-1,2, -1,step=1)
	k_ar_diff = st.number_input("Número de diferenciações", 0, 10, 0, step=1)
	cointegrar = st.button('Testar Cointegração')
	if cointegrar:
		if session_state.data is not None:
			cointegracao = coint_window(session_state.data, freq_, window_size_, approach_, variavel_y, variaveis_coint,det_order , k_ar_diff, nc)
			st.write(cointegracao) 

st.subheader('Modelos Temporais', help="ARIMA, SARIMA, VAR e VEC", divider='rainbow')				

st.subheader("Auto Arima", help ="Utiliza a função padrão autoarima para gerar um modelo. Baseado em AIC. Considera diferenciação quando d >= 1. Para usar um modelo com diferenciação aplique a função de diferenciação do dataframe. Você pode configurar a frequência do seu dataframe com as funções auxiliares")	

columns_list = [col for col in session_state.data.columns if col != 'Date']

a1, a2, a3= st.columns(3)

with(a1):
	variavel = st.selectbox("Selecione a Variável", columns_list)
	cut = st.number_input("Percentual de split treinamento e teste:", 0.0,1.0, 0.25,step=0.01)
	autoarima = st.button("auto_arima")
with(a2):
	max_p = st.number_input("Max_p:", 1,100,3,step=1)
	max_q = st.number_input("Max_q:", 1,100,3,step=1)
	d = st.number_input("d: None = -1", min_value=-1, max_value=100, step=1, value=-1, format="%d")
if d == -1:
    d = None

with(a3):
	m = st.number_input("Período da sazonalidade, False = 1",1,100,step=1)
	seasonal = st.selectbox("Permitir Sazonalidade:",[False, True])
	teste = st.radio("Teste", ["adf", "kpss", "pp"])

if autoarima:
	if session_state.data is not None:
		model = my_auto_arima(cut,session_state.data, variavel, teste,d, max_p,max_q,seasonal,m)

st.subheader("SARIMALL", help ="Você pode configurar os parâmetros individualmente ou deixar que o otimizador encontre o melhor modelo de acordo com um limite de variação dos parâmetros. Quando os parametros forem -1, significa que seguem o default None e a função usará análise combinatória para achar o melhor modelo. Certifique-se de garantir que você tem capacidade para processar a função. Se os parâmetros forem considerados, a função SARIMALL funciona como um SARIMA personalizado. Max_lags é o único parametro cujo default é variar até 12. Os demais variam até 3, para limitar o estouro de combinações. ATENÇÃO: TEMPO ESTIMADO É DE 5 SEGUNDOS POR MODELO. TEMPO MÉDIO MIL MODELOS É DE 13,8 HORAS ")	

st.markdown("Selecione os parametros para o otimizador")

s1, s2, s3= st.columns(3)

with s1:
	st.markdown("Configurações")
	variavel2 = st.selectbox("Selecione", columns_list)
	cut2 = st.number_input("Proporção de teste", 0.0,1.0, 0.25,step=0.01)
	metric = st.selectbox("Otimizar por:", ['rmse', 'mse', 'mae','mape','aic', 'bic'])
	n_plots = st.number_input("Quantidade de Plots otimizados",1,10,3,step=1)
	sarimall = st.button("SARIMALL")
with s2:
	st.markdown("Parâmetros") 
	p =st.number_input("p", -1,10,-1,step=1)
	d2 = st.number_input("d", -1,10,-1,step=1)
	q =st.number_input("q", -1,10,-1,step=1)
	limite_combinacoes = st.number_input("Limite p, d, q, P, D, Q:", 1,10,3,step=1)
	stationarity = st.selectbox("Forçar estacionariedade?", [True,False])
	
with s3:
	st.markdown("Parâmetros Sazonais") 					
	P =st.number_input("P", -1,10,-1,step=1)
	D =st.number_input("D", -1,10,-1,step=1)
	Q =st.number_input("Q",-1,10,-1,step=1)
	lags = st.number_input("Max Lags:", 2,30,12,step=1)
	variar_lag = st.selectbox("Lag fixo ou em range:", ["Fixo", "Range"])

if sarimall:
	if session_state.data is not None:
		model = SARIMALL(cut2, session_state.data, variavel2,stationarity, p, d2, q, P, D, Q,limite_combinacoes,lags,metric,variar_lag, n_plots)

st.subheader("GRANGER CAUSALIDADE", help ="Identifica quais variáveis X granger causam variável de interesse Y para um número de lags máximo. É preciso garantir que as séries estejam estacionárias, portanto número de diferenciações aplica a difença no conjunto de dados passado, mas não verifica estacionariedade. Para isso, use a função make stationar. Se VAR_SELEC for configurado para True, dataframe será modificado para preservar as colunas que granger causam a variável de interesse. Para isso, use a função make stationary")	

nc2 = st.selectbox("NCs:",[0.05,0.1, 0.01])
n_diff = st.number_input("Número de Diferenciações", 0,10,step=1)
mlags = st.number_input("Número de lags avaliados", 0,10,6,step=1)
variavelY = st.selectbox("Variavel Granger Causada?", session_state.data.columns)
op =  ["None"] + list(session_state.data.columns)
variaveisX = st.multiselect("Granger Causa", op)
VAR_SELECT = st.selectbox("Selecioanr Variáveis?", [False,True])

granger_causalidade = st.button("Causalidade de Granger")

if granger_causalidade:
    if session_state.data is not None:
        session_state.data = grangercausalitytests_trintinalia(session_state.data, variavelY, mlags, n_diff, nc2, variaveisX, VAR_SELECT)

st.subheader("AUTOVARVEC", help ="Utiliza análise combinatória para encontrar o VARVEC otimizado por parametros de previsão. Usa função Verificador de Coinregração. Se houver cointegração entre as séries configura um modelo VEC. Caso não haja cointegração, configura um modelo VAR combinatório que testa as n combinações para as colunas regressoras par a par")	


varY = st.selectbox("Variável para previsão", session_state.data.columns)
varX = st.multiselect("Lista de variáveis regressoras", session_state.data.columns)
vardiff = st.number_input("Diferenciações", 0,10,step=1)
varcut = st.number_input("split treinamento e teste:", 0.0, 1.0, 0.25, step=0.01)
var_lags = st.number_input("Var Lags:", 1, 100, 6, step=1)
autovar = st.button("AutoVAR")
if autovar:
    if session_state.data is not None:
        AUTOVAR(session_state.data,vardiff,varcut,var_lags, varY, varX)


st.subheader('Betas', help="Calcula e plota Betas entre ativos selecionados", divider='rainbow')			

colunas_adj_close = [col for col in session_state.data.columns if col.startswith('Adj Close')]

betas = st.multiselect("Colunas:", colunas_adj_close)
calcular_betas = st.button("calcular betas")
if calcular_betas:
	if session_state.data is not None:
		gerar_betas(session_state.data, betas)

st.subheader('Chat', help="Deixe uma mensagem", divider='rainbow')			

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Exibe as mensagens do histórico quando o aplicativo é reiniciado
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Deixe uma mensagem!"):
    # Exibe a mensagem do usuário no container de mensagens do chat
    with st.chat_message("user"):
        st.markdown(prompt)
    # Adiciona a mensagem do usuário ao histórico usando a função chat()
    chat(prompt, "user", st.session_state.messages)

    # Mantém o histórico de chat dentro do limite máximo
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages.pop(0)  # Remove a mensagem mais antiga
    # Salva o histórico no arquivo
    save_chat_history(st.session_state.messages)
