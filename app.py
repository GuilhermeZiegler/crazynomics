##  Importação do streamlit para exibição da Dashboard
import streamlit as st

## Importação de bibliotecas de manipulação padrão
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta, datetime, time

## importação de bibliotecas de plotagem de visualziação de dados
import seaborn as sns
import matplotlib.pyplot as plt

## importação de bibliotecas de strings
import io
import requests
import warnings
import copy

import unidecode
import re
import string
import joblib
import glob


## importações de modelos
import optuna
from category_encoders import JamesSteinEncoder, WOEEncoder, CatBoostEncoder
from sklearn.impute import SimpleImputer
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

def tickerget(ticker_df, start, end):
    df_list = []

    for index, row in ticker_df.iterrows():
        key = index  # Primeira coluna contendo a chave
        ticker = row[0]  # Segunda coluna contendo o ticker

        dados = yf.download(ticker, start=start, end=end)
        dados['Ticker'] = key  # Adiciona uma coluna 'Ticker' com o símbolo ou código
        dados['Key'] = key  # Adiciona uma coluna 'Key' com a chave
        df_list.append(dados)

    df_data = pd.concat(df_list)
    return df_data

def split_dataframes_by_ticker(df):
    grouped_data = df.groupby('Ticker')

    dataframes = {}

    for ticker, group in grouped_data:
        new_columns = [f'{column}_{ticker}' for column in group.columns]
        group.columns = new_columns
        dataframes[ticker] = group.copy()

    return dataframes

def merge_dataframes_by_date(dataframes_list):
    merged_df = pd.concat(dataframes_list, axis=1)
    merged_df.index = pd.to_datetime(merged_df.index)  # Certifique-se de que o índice seja do tipo de data
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]  # Remova linhas duplicadas, se houver
    return merged_df

def BaixaYahoo(df, start, end):
    df_data = tickerget(df, start, end)
    dataframes = split_dataframes_by_ticker(df_data)
    df_concatenated =  merge_dataframes_by_date(dataframes)
    return df_data, dataframes, df_concatenated


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

# Função para preencher NAs com médias na janela
def fill_moving_avg(df, window_size):
    for col in df.select_dtypes(include=[np.number]).columns:  # Seleciona apenas colunas numéricas
        df[col] = df[col].rolling(window=window_size, min_periods=1, center=False).mean()

def puxa_dados(merged_df, start, end):
    _, _, dfs = BaixaYahoo(merged_df, start, end)
    dfs.columns = dfs.columns.droplevel()
    st.write('Dados baixados:', dfs.shape)
    st.write('Quantidade de NaN:', dfs.isna().sum().to_frame().T)
    
    first_index = firstrow_notna(dfs)
    last_index = lastrow_notna(dfs)
    first_ind = pd.to_datetime(first_index)
    last_ind = pd.to_datetime(last_index)
    
    if first_index is not None and pd.to_datetime(first_ind) > pd.to_datetime(start_date):
        st.warning(f'Data de início ótima: {first_index}', icon="⚠️")
    
    if last_index is not None and pd.to_datetime(last_ind) < pd.to_datetime(end_date):
        st.warning(f'Data de término ótima: {last_index}', icon="⚠️")
    
    dfs.drop(dfs[(dfs.index < first_ind) | (dfs.index > last_ind)].index, inplace=True)
    return dfs

def heavycleaning(dfs, limpeza_pesada):
    for prefix in limpeza_pesada:
        columns_to_drop = [col for col in dfs.columns if col.startswith(prefix)]
        dfs.drop(columns=columns_to_drop, inplace=True)
    st.write('Quantidade de NaN:', dfs.isna().sum().to_frame().T)
    st.write("Tamanho após a limpeza Pesada: ", dfs.shape)
    return dfs 

def cortar_volume(dfs, corte_volume):
    columns_to_drop = [col for col in dfs.columns if col.startswith('Volume_') and (dfs[col] == 0).sum() >= corte_volume]
    dfs.drop(columns=columns_to_drop, inplace=True)
    st.write('Quantidade de NaN:', dfs.isna().sum().to_frame().T)
    st.write("Tamanho após corte de % Volumes Zerados na coluna:", dfs.shape)
    return dfs


def fill_moving_avg(df, window_size):
    for col in df.select_dtypes(include=[np.number]).columns:  # Seleciona apenas colunas numéricas
        df[col] = df[col].rolling(window=window_size, min_periods=1, center=False).mean()

    st.write("Tamanho após aplicar médias móveis:", df.shape)
    st.write(f'Média Móvel em dias: {window_size}')
    st.write(f'Quantidade de NaN: {df.isna().sum().sum()} dados nulos')
    st.write(df.isna().sum().to_frame().T)


## Configuração da página e do título
st.set_page_config(page_title='Monografia Guilherme Ziegler', layout = 'wide', initial_sidebar_state = 'auto')

st.title("Previsão do preço da soja via LSTM e seletor automatizado de modelo VARVEC por meio de aplicativo interativo")
        
# Importar SessionState
class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

@st.cache(allow_output_mutation=True)
def get_session():
    return SessionState(data=None, cleaned=False)

session_state = get_session()


## Bloco de datas para slicer

max_date = dt.datetime.today().date()
min_date = (dt.datetime.today() - dt.timedelta(days=10 * 700)).date()

default_start = '2007-07-20'
default_end = '2023-08-22'

default_start = dt.datetime.strptime(default_start, '%Y-%m-%d').date()
default_end = dt.datetime.strptime(default_end, '%Y-%m-%d').date()

selected_range = st.slider("Selecione o intervalo de datas", min_date, max_date, (default_start, default_end))
start_date = selected_range[0]
end_date = selected_range[1]
st.write(f"Dados para o range de datas selecionado: {start_date} até {end_date}")

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
st.info("Moedas atualmente disponíveis para download")
default_moedas = moedas_mono.keys()
moedas_keys = list(moedas.keys())
moedas_selecionadas = st.multiselect("Moedas sugeridas na Monografia:", moedas_keys,default= default_moedas )
if moedas_selecionadas:
    # Criar um DataFrame a partir do dicionário filtrando as moedas selecionadas
    df_moedas = pd.DataFrame(moedas.items(), columns=['Moeda', 'Ticker']).set_index('Moeda').loc[moedas_selecionadas]
    # Exibir o DataFrame resultante
    #st.write(df_moedas)
else:
    st.info("Selecione pelo menos uma moeda para continuar.")

## Bloco de commodities
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
st.info("Commodities atualmente disponíveis para download")
commodities_keys = list(lista_commodities.keys())
commodities_selecionadas = st.multiselect("Commodities sugeridas na Monografia:", commodities_keys,default= default_commodities )
if commodities_selecionadas:
    # Criar um DataFrame a partir do dicionário filtrando as moedas selecionadas
    df_commodities = pd.DataFrame(lista_commodities.items(), columns=['Commodities', 'Ticker']).set_index('Commodities').loc[commodities_selecionadas]
        # Exibir o DataFrame resultante
        #st.write(df_commodities)
else:
    st.info("Selecione pelo menos uma commoditie para continuar")

## Bloco de empresas
empresas_mono = {
                'MARFRIG': 'MRFG3.SA',
                'BRF': 'BRFS3.SA',
                'MINERVA': 'BEEF3.SA',
                'JBS': 'JBSS3.SA',
                'AGRO3':'AGRO3.SA'
                }

lista_empresas = {
                'AGRO3': 'AGRO3.SA',
                'ALPARGATAS': 'ALPA4.SA',
                'AMBEV S/A': 'ABEV3.SA',
                'AMERICANAS': 'AMER3.SA',
                'ASML Holding': 'ASML34.SA',
                'ASSAI': 'ASAI3.SA',
                'AZUL': 'AZUL4.SA',
                'B3': 'B3SA3.SA',
                'BANCO PAN': 'BPAN4.SA',
                'BBSEGURIDADE': 'BBSE3.SA',
                'BRADESCO': 'BBDC4.SA',
                'BRADESPAR': 'BRAP4.SA',
                'BRASIL': 'BBAS3.SA',
                'BRASKEM': 'BRKM5.SA',
                'BRF': 'BRFS3.SA',
                'BRF SA': 'BRFS3.SA',
                'BTGP BANCO': 'BPAC11.SA',
                'Bemobi Mobile Tech SA': 'BMOB3.SA',
                'Broadcom Inc.': 'AVGO34.SA',
                'CARREFOUR BR': 'CRFB3.SA',
                'CCR SA': 'CCRO3.SA',
                'CEMIG': 'CMIG4.SA',
                'CIELO': 'CIEL3.SA',
                'COGNA ON': 'COGN3.SA',
                'COPEL': 'CPLE6.SA',
                'COSAN': 'CSAN3.SA',
                'CPFL ENERGIA': 'CPFE3.SA',
                'CVC BRASIL': 'CVCB3.SA',
                'CYRELA REALT': 'CYRE3.SA',
                'DEXCO': 'DXCO3.SA',
                'Direcional Engenharia S.A.': 'DIRR3.SA',
                'ECORODOVIAS': 'ECOR3.SA',
                'ELETROBRAS': 'ELET6.SA',
                'EMBRAER': 'EMBR3.SA',
                'ENERGIAS BR': 'ENBR3.SA',
                'ENERGISA': 'ENGI11.SA',
                'ENEVA': 'ENEV3.SA',
                'ENGIE BRASIL': 'EGIE3.SA',
                'EQUATORIAL': 'EQTL3.SA',
                'EZTEC': 'EZTC3.SA',
                'Enjuei': 'ENJU3.SA',
                'FLEURY': 'FLRY3.SA',
                'GERDAU': 'GGBR4.SA',
                'GERDAU MET': 'GOAU4.SA',
                'GOL': 'GOLL4.SA',
                'GRUPO NATURA': 'NTCO3.SA',
                'HAPVIDA': 'HAPV3.SA',
                'HYPERA': 'HYPE3.SA',
                'IRBBRASIL RE': 'IRBR3.SA',
                'ITAUSA': 'ITSA4.SA',
                'ITAUUNIBANCO': 'ITUB4.SA',
                'JBS': 'JBSS3.SA',
                'JHSF PART': 'JHSF3.SA',
                'KLABIN S/A': 'KLBN11.SA',
                'LOCALIZA': 'RENT3.SA',
                'LOCAWEB': 'LWSA3.SA',
                'LOJAS RENNER': 'LREN3.SA',
                'M. Dias Branco': 'MDIA3.SA',
                'MAGAZ LUIZA': 'MGLU3.SA',
                'MARFRIG': 'MRFG3.SA',
                'MELIUZ': 'CASH3.SA',
                'MINERVA': 'BEEF3.SA',
                'MRV': 'MRVE3.SA',
                'MULTIPLAN': 'MULT3.SA',
                'Microsoft BDR': 'MSFT34.SA',
                'P.ACUCAR-CBD': 'PCAR3.SA',
                'PETROBRAS': 'PETR4.SA',
                'PETRORIO': 'PRIO3.SA',
                'PETZ': 'PETZ3.SA',
                'Plascar Participacoes Industriais SA': 'PLAS3.SA',
                'QUALICORP': 'QUAL3.SA',
                'RAIADROGASIL': 'RADL3.SA',
                'REDE D OR': 'RDOR3.SA',
                'RUMO S.A.': 'RAIL3.SA',
                'SABESP': 'SBSP3.SA',
                'SANTANDER BR': 'SANB11.SA',
                'SID NACIONAL': 'CSNA3.SA',
                'SUZANO S.A.': 'SUZB3.SA',
                'TAESA': 'TAEE11.SA',
                'TELEF BRASIL': 'VIVT3.SA',
                'TIM': 'TIMS3.SA',
                'TOTVS': 'TOTS3.SA',
                'Taiwan Semiconduc Manufact Co Lt Bdr': 'TSMC34.SA',
                'ULTRAPAR': 'UGPA3.SA',
                'USIMINAS': 'USIM5.SA',
                'VALE': 'VALE3.SA',
                'VIA': 'VIIA3.SA',
                'WEG': 'WEGE3.SA',
                'YDUQS PART': 'YDUQ3.SA'
                }

st.info("Empresas diponíveis para download")
default_empresas = list(empresas_mono.keys())
empresas_keys = list(lista_empresas.keys())
empresas_selecionadas = st.multiselect("Empresas sugeridas na Monografia:", empresas_keys,default= default_empresas )
if empresas_selecionadas:
    # Criar um DataFrame a partir do dicionário filtrando as moedas selecionadas
    df_empresas = pd.DataFrame(lista_empresas.items(), columns=['Empresa', 'Ticker']).set_index('Empresa').loc[empresas_selecionadas]
        # Exibir o DataFrame resultante
        #st.write(df_empresas)
else:
    st.info("Selecione pelo menos uma empresa para continuar")

## Bloco de commodities
indices_mono  = {'S&P GSCI':'GD=F', #  commodities agrícolas, incluindo soja, trigo, milho e algodão
                'IBOVESPA': '^BVSP',
               }

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
                'EGX 30 Price Return Index': '^CASE30',
                'Top 40 USD Net TRI Index': '^JN0U.JO',
                'NIFTY 50': '^NSEI'
                }

st.info("Indices diponíveis para download")            
default_indices = list(indices_mono.keys())
indices_keys = list(lista_indices.keys())
indices_selecionados = st.multiselect("Indices sugeridos na Monografia:", indices_keys,default= default_indices )
if indices_selecionados:
    # Criar um DataFrame a partir do dicionário filtrando as moedas selecionadas
    df_indices = pd.DataFrame(lista_indices.items(), columns=['Indice', 'Ticker']).set_index('Indice').loc[indices_selecionados]
    # Exibir o DataFrame resultante
    #st.write(df_indices)
else:
    st.info("Selecione pelo menos uma indice para continuar")

## Funções para baixar os dados 

## Bloco de seleç]ao de variáveis
st.subheader("Variáveis selecionadas:") 
merged_df = pd.concat([df_moedas, df_empresas, df_commodities, df_indices], axis=0)
st.markdown(merged_df.index.to_list())
start = start_date 
end = end_date

## bloco de limpeza pesada para tickers
prefixes = ['Open_', 'Close_','High_', 'Low_', 'Adj Close_', 'Volume_', 'Ticker_', 
            'Key_', 'Diff_daily_', 'Amplitude_', 'Retorno_daily_', 
            'MAX_Amplitude_Change_', 'Normal_Amplitude_Change_','Updown']

limpeza_pesada = st.sidebar.multiselect('Remova colunas com a estrutura NOME_',prefixes, default=('Key_','Ticker_'))

if st.button("Puxar Dados"):
    session_state.data = puxa_dados(merged_df, start_date, end_date)

if st.sidebar.button("Fazer Limpeza"):
    if session_state.data is not None:
        session_state.data = heavycleaning(session_state.data, limpeza_pesada)

# Exibição dos resultados
if session_state.data is not None:
    st.write("DataFrame:")
    st.dataframe(session_state.data)

## bloco de corte por volume por percentual de zeros

corte_volume = st.sidebar.slider('Remove Volume_ para percentual de 0 na coluna', 0, 100, 100, step=1)

if st.sidebar.button("Cortar Volume"):
    if session_state.data is not None:
        cortar_volume(session_state.data, corte_volume)

dias_moving_avg =  st.sidebar.slider('Dias para inputar média móvel:',1, 100, 20,step=1)                     

if st.sidebar.button("Alicar Moving Avg"):
    if session_state.data is not None:
            fill_moving_avg(session_state.data, dias_moving_avg)

baixar_excel = st.button("Baixar Excel")
if baixar_excel:
    if session_state.data is not None:
        dfs_rounded = session_state.data.round(6)
        
        # Crie uma cópia do DataFrame para manter o índice original
        dfs_copy = dfs_rounded.copy()
        
        excel_file = "dados.xlsx"  # Nome do arquivo Excel a ser criado
        with st.spinner("Criando arquivo Excel..."):
            dfs_copy.to_excel(excel_file, index=True, header=True)
        st.success(f"Arquivo Excel criado com sucesso! Clique no botão abaixo para baixar.")
        st.download_button(
            label="Baixar dados em Excel",
            data=open(excel_file, "rb").read(),
            file_name=excel_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
