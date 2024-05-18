import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Carregar o arquivo CSV
@st.cache_data
def df():
    tabela = pd.read_csv('https://raw.githubusercontent.com/TrabalhoProjetado/Tech_Challenger/main/Base%20Petro%CC%81leo.csv', delimiter=';', parse_dates=['Data'], dayfirst=True)
    tabela['Preço'] = tabela['Preço - petróleo bruto - Brent (FOB)'].str.replace(',', '.').astype(float)
    return tabela

# Criar a interface do Streamlit

st.set_page_config(
    page_title='Tech Challenger 4',
    page_icon = "🛢️",
    layout="wide",
    initial_sidebar_state="auto",#"expanded"
)

#Titulo da pagina
st.title('Modelo de Produção de Preços do Petróleo Brent')


# Criar os botões e links
botoes = ["Ir para o Modelo ML", "Ir para o Dashboard", "Ir para o Github"]
links = ["https://colab.research.google.com/drive/19UHI53aHKVm-Nv-6n5n-nyyVNdvsU6l9?usp=sharing#scrollTo=FYS1NRzdI3-K","https://app.powerbi.com/view?r=eyJrIjoiNTVhNzIyODgtNjRkNy00NDIxLWFlZWQtZWY0YWRlNmY1NjU1IiwidCI6IjdmZTFiYmJjLWIwMmUtNDFmMS04N2YyLTNhNWIzMTY1NzM0ZiJ9 ", "https://github.com/TrabalhoProjetado/Tech_Challenger"] #ADICIONAR POWER BI

st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #D64550;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

colu1, colu2, colu3 = st.columns(3)
with colu1:
    if st.button(botoes[0]):
        st.markdown(f'[{botoes[0]}]({links[0]})', unsafe_allow_html=True)

with colu2:
    if st.button(botoes[1]):
        st.markdown(f'[{botoes[1]}]({links[1]})', unsafe_allow_html=True)

with colu3:
    if st.button(botoes[2]):
        st.markdown(f'[{botoes[2]}]({links[2]})', unsafe_allow_html=True)


# Obter o preço mais recente do petróleo
preco_mais_recente = df().iloc[0]['Preço - petróleo bruto - Brent (FOB)']
media_preco = df()['Preço'].mean()

#informações 
col1, col2 = st.columns(2)
with col1:
    st.write(f"""
        <p style="font-size: 25px;">
            Último Valor do petróleo:
        </p>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown(
            f"""
            <div style="
                background-color: rgba(194, 241, 200, 0.0);
                padding: 10px;
                border-radius: 1px;
                text-align: center;
            ">
                <p style="
                    font-size: 30px; 
                    color: #FFFFFF;
                    font-weight: bold;
                    ">{preco_mais_recente}$
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
    st.write(f"""
        <p style="font-size: 25px;">
            Média preço do petróleo:
        </p>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown(
            f"""
            <div style="
                background-color: rgba(194, 241, 200, 0.0);
                padding: 10px;
                border-radius: 1px;
                text-align: center;
            ">
                <p style="
                    font-size: 30px; 
                    color: #FFFFFF;
                    font-weight: bold;
                    ">{media_preco:.2f}$
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# Grafico

st.write(f"""
    <p style="font-size: 24px;">
        Analise por dias :
    </p>
""", unsafe_allow_html=True)

quant_dias = st.selectbox("Selecione o período de filtro", ["7 dias", "15 dias", "30 dias"])
num_dias = int(quant_dias.replace(" dias", ""))

# Filtro do Dataframe Baseado na seleção
graf = df().head(num_dias)

# Cria o Grafico de area
fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(graf['Data'], graf['Preço - petróleo bruto - Brent (FOB)'], color='skyblue', alpha=0.4)
ax.plot(graf['Data'], graf['Preço - petróleo bruto - Brent (FOB)'], color='Slateblue', alpha=0.6)
ax.set_xlabel('Data')
ax.set_ylabel('Preço (dólares)')
ax.set_title(f'Preços do Petróleo Brent nos últimos {num_dias} dias')
ax.grid(True)

# Mostrar no Streamlit
st.pyplot(fig)


#---------------------------------------------------------------------------------------------------------------------
#BASE ANALITICA
st.write(f"""
    <p style="font-size: 20px;">
      Exportação da Base Analítica:
    </p>
""", unsafe_allow_html=True)
#Filtrar e Baixar a base Analitica
min_data = df()["Data"].min()
max_data = df()["Data"].max()

inicio = min_data
fim = max_data

inicio = st.date_input("Seleciona a data de Inicio", min_data, min_value=min_data, max_value=max_data)
fim = st.date_input("Seleciona a data Final", max_data, min_value=min_data, max_value=max_data)

inicio =pd.to_datetime(inicio)
fim =pd.to_datetime(fim)

if inicio and fim and inicio < fim:
    filtro_data =df()[(df()["Data"]>= inicio)&(df()["Data"]<= fim)]
    st.write("Filtro do Dataframe")
    st.dataframe(filtro_data.iloc[:, :2])
else:
    st.error("porfavor escolha uma Data Valida")

