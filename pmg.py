import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO

# Criar a interface do Streamlit

st.set_page_config(
    page_title='Passos Mágicos',
    page_icon = "https://img.icons8.com/?size=100&id=SS9irMJVAaEa&format=png&color=000000", #Icone de pegada
    layout="wide",
    initial_sidebar_state="auto",#"expanded"
)

#Titulo da pagina
st.markdown("""
<style>
h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 3em;'>ASSOCIAÇÃO PASSOS MÁGICOS</h1>", unsafe_allow_html=True)


# Criar os botões e links

botoes = ["Ir para o Dashboard", "Ir para o Github","Ir pro Colab"]
links = ["https://app.powerbi.com/view?r=eyJrIjoiZmI5OTAwZWItZGFlMS00NTk3LWJlOWUtZjcwMWQzYzI1N2NkIiwidCI6IjdmZTFiYmJjLWIwMmUtNDFmMS04N2YyLTNhNWIzMTY1NzM0ZiJ9", "https://github.com/TrabalhoProjetado/Passos-Magicos-","https://colab.research.google.com/drive/1QbxXfVYMFBcRwM8aiVfIA3TauDL3jN1b?usp=sharing"] #ADICIONAR POWER BI E COLAB


st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #E99312;
        color: #FFFFFF;
        padding: 10px 20px;
        font-size: 15px; 
        
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

#Carregar a Base
@st.cache_data
def load_data():
    tabela = pd.read_csv('https://raw.githubusercontent.com/TrabalhoProjetado/Passos-Magicos-/main/Base_Final_1208.csv', delimiter=',', dayfirst=True)
    return tabela
# Carregar os dados
data = load_data()

st.write("")
st.write("")

#Subtítulo
st.markdown("<h1 style='color: white; font-size: 2em; text-align: left; margin-left: 0;'><u>Quantidade de Alunos por categoria em 2022</u></h1>", unsafe_allow_html=True)

# Filtrar os dados para 2022
filtered_data = data[(data['PEDRA_2022'] == 'Topázio') & (data['PEDRA_2022'] != '#NULO!')]
quantidade_topazio = filtered_data.shape[0] 

filtered2_data = data[(data['PEDRA_2022'] == 'Ametista') & (data['PEDRA_2022'] != '#NULO!')]
quantidade_ametista = filtered2_data.shape[0] 

filtered3_data = data[(data['PEDRA_2022'] == 'Ágata') & (data['PEDRA_2022'] != '#NULO!')]
quantidade_agata = filtered3_data.shape[0]  

filtered4_data = data[(data['PEDRA_2022'] == 'Quartzo') & (data['PEDRA_2022'] != '#NULO!')]
quantidade_quartzo = filtered4_data.shape[0] 

# Exibir a quantidade no Streamlit
col1, col2, col3, col4 = st.columns(4)  # Cria quatro colunas

with col1:
    st.markdown(f"<h1 style='color: #2823bc; font-size: 1.8em;'>Topázio é: <span style='color: white;'>{quantidade_topazio}</span></h1>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<h1 style='color: #800080; font-size: 1.8em;'>Ametista é: <span style='color: white;'>{quantidade_ametista}</span></h1>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<h1 style='color: #fc5500; font-size: 1.8em;'>Ágata é: <span style='color: white;'>{quantidade_agata}</span></h1>", unsafe_allow_html=True)

with col4:
    st.markdown(f"<h1 style='font-size: 1.8em;'>Quartzo é: <span style='color: white;'>{quantidade_quartzo}</span></h1>", unsafe_allow_html=True)

#GRAFICOS

st.write("")
                                                #BARRAS
st.write("")

# Definir os bins e os anos
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
years = ['2020', '2021', '2022']
binned_data = {year: [] for year in years}

# Preparar os dados
for year in years:
    data['IPV_' + year] = pd.to_numeric(data['IPV_' + year], errors='coerce')
    binned_series = pd.cut(data['IPV_' + year], bins=bins, right=True)
    binned_counts = binned_series.value_counts().reindex(pd.IntervalIndex.from_breaks(bins), fill_value=0).sort_index()
    binned_data[year] = binned_counts

# Criar o gráfico
fig = go.Figure()

bar_width = 0.2
for i, year in enumerate(years):
    fig.add_trace(go.Bar(
        x=[str(interval) for interval in binned_data[year].index],
        y=binned_data[year],
        name=year,
        width=bar_width,
        offset=-bar_width + (i * bar_width),
        marker_color=['#0000CD', '#FF4500', '#006400'][i],  # Paleta de cores
        hoverlabel=dict(bgcolor='rgba(207,207,209,1)', font_color='black')  # Cor de fundo do tooltip
    ))

# Ajustar layout do gráfico
fig.update_layout(
    title='Distribuição do INDE (2020, 2021, 2022)',
    xaxis_title='INDE',
    yaxis_title='Frequência',
    barmode='group',
    bargap=0.1,  # Espaço entre as barras
    bargroupgap=0.1,  # Espaço entre as barras do grupo
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',  
    title_font=dict(size=30),  # Tamanho do Título
    xaxis_title_font=dict(size=20),  # Tamanho do Eixo x
    yaxis_title_font=dict(size=20),   # Tamanho do Eixo y
    legend_font=dict(size=20),  # Tamanho da Legenda
    width=1300,  # Tamanho da Largura
    height=900  # Tamanho da Altura
)

# Exibir o gráfico no Streamlit
st.title("Distribuição do INDE por Ano")
st.write('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)
st.write('</div>', unsafe_allow_html=True)


st.write("")
                                                #RADAR
st.write("")

# Calcular a média das pedras para cada instituição em 2020, 2021 e 2022
instituicoes_2020_avg = data.groupby('INSTITUICAO_ENSINO_ALUNO_2020')['PEDRA_NUM'].mean().dropna()
instituicoes_2021_avg = data.groupby('INSTITUICAO_ENSINO_ALUNO_2021')['PEDRA_NUM'].mean().dropna()
instituicoes_2022_avg = data.groupby('INSTITUICAO_ENSINO_ALUNO_2022')['PEDRA_NUM'].mean().dropna()

# Função para plotar gráfico de radar
def plot_radar(data, title, color):
    labels = data.index.tolist()
    stats = data.values

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=stats,
        theta=labels + [labels[0]],
        fill='toself',
        name=title,
        fillcolor=color['fillcolor'],
        line=dict(color=color['linecolor'], width=2)
    ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(stats) + 1],
                tickfont=dict(color='black'),
            )
        ),
        showlegend=False
    )

    return fig

# Definindo cores para cada ano
colors = {
    '2020': {'fillcolor': 'rgba(0, 0, 255, 0.25)', 'linecolor': 'blue'},
    '2021': {'fillcolor': 'rgba(255, 165, 0, 0.25)', 'linecolor': 'orange'},
    '2022': {'fillcolor': 'rgba(0, 128, 0, 0.25)', 'linecolor': 'green'}
}

# Subplots para os gráficos
fig_2020 = plot_radar(instituicoes_2020_avg, 'Média das Pedras por Instituição em 2020', colors['2020'])
fig_2021 = plot_radar(instituicoes_2021_avg, 'Média das Pedras por Instituição em 2021', colors['2021'])
fig_2022 = plot_radar(instituicoes_2022_avg, 'Média das Pedras por Instituição em 2022', colors['2022'])

# Exibir no Streamlit
st.title("Gráficos de Radar - Média das Pedras por Instituição")

col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(fig_2020, use_container_width=True)

with col2:
    st.plotly_chart(fig_2021, use_container_width=True)

with col3:
    st.plotly_chart(fig_2022, use_container_width=True)


st.write("")
                                                #BOLHAS
st.write("")

# Converter colunas INDE para numérico
data['INDE_2020'] = pd.to_numeric(data['INDE_2020'], errors='coerce')
data['INDE_2021'] = pd.to_numeric(data['INDE_2021'], errors='coerce')
data['INDE_2022'] = pd.to_numeric(data['INDE_2022'], errors='coerce')

# Calcular métricas
def calcular_metricas(df, ano_col, inde_col):
    num_alunos = df[ano_col].value_counts()
    media_inde = df.groupby(ano_col)[inde_col].mean()
    retencao = df[ano_col].count() / df[ano_col].count()
    return num_alunos, media_inde, retencao

# Gerando os dados para 2020, 2021 e 2022
num_alunos_2020, media_inde_2020, retencao_2020 = calcular_metricas(data, 'INSTITUICAO_ENSINO_ALUNO_2020', 'INDE_2020')
num_alunos_2021, media_inde_2021, retencao_2021 = calcular_metricas(data, 'INSTITUICAO_ENSINO_ALUNO_2021', 'INDE_2021')
num_alunos_2022, media_inde_2022, retencao_2022 = calcular_metricas(data, 'INSTITUICAO_ENSINO_ALUNO_2022', 'INDE_2022')

# Criar DataFrames para cada ano
df_2020 = pd.DataFrame({'Instituição': num_alunos_2020.index, 'Número de Alunos': num_alunos_2020.values, 'Média do INDE': media_inde_2020.values, 'Retenção': retencao_2020})
df_2021 = pd.DataFrame({'Instituição': num_alunos_2021.index, 'Número de Alunos': num_alunos_2021.values, 'Média do INDE': media_inde_2021.values, 'Retenção': retencao_2021})
df_2022 = pd.DataFrame({'Instituição': num_alunos_2022.index, 'Número de Alunos': num_alunos_2022.values, 'Média do INDE': media_inde_2022.values, 'Retenção': retencao_2022})

# Criar gráfico de bolhas
fig_2020 = px.scatter(df_2020, x='Número de Alunos', y='Média do INDE', size='Retenção', color='Instituição', title='Análise das Instituições (2020)')
fig_2021 = px.scatter(df_2021, x='Número de Alunos', y='Média do INDE', size='Retenção', color='Instituição', title='Análise das Instituições (2021)')
fig_2022 = px.scatter(df_2022, x='Número de Alunos', y='Média do INDE', size='Retenção', color='Instituição', title='Análise das Instituições (2022)')

# Exibir no Streamlit
st.title("Análise das Instituições por Ano")
col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(fig_2020, use_container_width=True)

with col2:
    st.plotly_chart(fig_2021, use_container_width=True)

with col3:
    st.plotly_chart(fig_2022, use_container_width=True)

# Ajustar o tamanho
st.write('<style>div.row {flex-direction: row;}</style>', unsafe_allow_html=True)

st.write("")
                                                #EXPORTAÇÃO DE BASE
st.write("")

# Exibir um exemplo da base
st.title("Exemplo da Base de Dados")
st.dataframe(data.head(5))  # Mostra 5 linhas da base

# Exportar a base de dados
st.markdown("### Exportar Base de Dados")
st.download_button(
    label="Download da Base de Dados",
    data=data.to_csv(index=False).encode('utf-8'),
    file_name='Base_Final.csv',
    mime='text/csv',
)