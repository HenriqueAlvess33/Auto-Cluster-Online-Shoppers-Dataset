# Importa as bibliotecas necessárias para análise de dados e visualização
import pandas as pd  # Manipulação de dados em DataFrames
import matplotlib.pyplot as plt  # Criação de gráficos
import seaborn as sns  # Visualização de dados estatísticos
import numpy as np  # Operações numéricas e manipulação de arrays
import streamlit as st
import io

from sklearn.decomposition import PCA  # Análise de Componentes Principais
from sklearn.preprocessing import StandardScaler  # Normalização de dados
from sklearn.cluster import KMeans  # Algoritmo de agrupamento K-Means
from sklearn.metrics import silhouette_score  # Métrica de avaliação de agrupamentos
from sklearn.utils import resample


def main():
    # Configura o título da aplicação
    st.set_page_config(
        page_title="Projeto para criação de clusters",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="varig_icon.png",
    )
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

    <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 3em;'>
        <strong>Clusterização do conjunto de dados</strong>
    </h1>
    """,
        unsafe_allow_html=True,
    )


# Função para plotar o scree plot e determinar o número ideal de componentes principais
@st.cache_data
def calcular_info_screeplot(
    explained_variance,
    explained_variance_ratio,
    n_components,
    ncomp=0,
    varexplicada=0,
    criterio=1,
):
    if ncomp > 0:
        ncomp_crit = ncomp
    elif varexplicada > 0:
        ncomp_crit = (explained_variance_ratio.cumsum() < varexplicada).sum() + 1
    elif criterio == 1:
        ncomp_crit = (explained_variance_ratio > 1 / n_components).sum()
    else:
        ncomp_crit = None

    variancia = explained_variance[ncomp_crit - 1]
    variancia_acumulada = explained_variance.cumsum()[ncomp_crit - 1]
    pct_variancia = explained_variance_ratio[ncomp_crit - 1]
    pct_variancia_acumulada = explained_variance_ratio.cumsum()[ncomp_crit - 1]

    return {
        "ncomp_crit": ncomp_crit,
        "variancia": variancia,
        "variancia_acumulada": variancia_acumulada,
        "pct_variancia": pct_variancia,
        "pct_variancia_acumulada": pct_variancia_acumulada,
    }


def plotagem_de_graficos_silhueta(
    explained_variance_, explained_variance_ratio_, n_components_, ncomp_crit
):
    # Cria uma grade de subplots para os gráficos
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(14, 8))
    plt.subplots_adjust(hspace=0, wspace=0.15)

    # Eixo X: número dos componentes principais
    num_componentes = np.arange(n_components_) + 1

    # Gráfico da variância explicada por componente
    ax[0, 0].plot(
        num_componentes,
        explained_variance_,
        "o-",
        linewidth=2,
        color="blue",
        markersize=2,
        alpha=0.2,
    )
    ax[0, 0].set_title("Scree Plot - Variância total")
    ax[0, 0].set_xlabel("Número de componentes")
    ax[0, 0].set_ylabel("Variancia explicada (Autovalores)")

    # Gráfico da variância explicada acumulada
    ax[1, 0].plot(
        num_componentes,
        explained_variance_.cumsum(),
        "o-",
        linewidth=2,
        color="blue",
        markersize=2,
        alpha=0.2,
    )
    ax[1, 0].set_xlabel("Número de componentes")
    ax[1, 0].set_ylabel("Variancia explicada (Acumulada)")

    # Gráfico da variância percentual explicada por componente
    ax[0, 1].plot(
        num_componentes,
        explained_variance_ratio_,
        "o-",
        linewidth=2,
        color="blue",
        markersize=2,
        alpha=0.2,
    )
    ax[0, 1].set_title("Scree Plot - Variância percentual")
    ax[0, 1].set_xlabel("Número de componentes")
    ax[0, 1].set_ylabel("Variancia explicada (percentual)")

    # Gráfico da variância percentual acumulada
    ax[1, 1].plot(
        num_componentes,
        explained_variance_ratio_.cumsum(),
        "o-",
        linewidth=2,
        color="blue",
        markersize=2,
        alpha=0.2,
    )
    ax[1, 1].set_xlabel("Número de componentes")
    ax[1, 1].set_ylabel("Variancia explicada (% Acumulado)")

    if ncomp_crit != None:
        # Linhas verticais de referência para o número de componentes escolhido
        ax[0, 0].axvline(x=ncomp_crit, color="r", linestyle="-", linewidth=0.5)
        ax[1, 1].axvline(x=ncomp_crit, color="r", linestyle="-", linewidth=0.5)
        ax[1, 0].axvline(x=ncomp_crit, color="r", linestyle="-", linewidth=0.5)
        ax[0, 1].axvline(x=ncomp_crit, color="r", linestyle="-", linewidth=0.5)

        # Linhas horizontais de referência para os valores correspondentes ao componente escolhido
        variancia = explained_variance_[ncomp_crit - 1]
        variancia_acumulada = explained_variance_.cumsum()[ncomp_crit - 1]
        pct_variancia = explained_variance_ratio_[ncomp_crit - 1]
        pct_variancia_acumulada = explained_variance_ratio_.cumsum()[ncomp_crit - 1]

        ax[0, 0].axhline(y=variancia, color="r", linestyle="-", linewidth=0.5)
        ax[1, 0].axhline(y=variancia_acumulada, color="r", linestyle="-", linewidth=0.5)
        ax[0, 1].axhline(y=pct_variancia, color="r", linestyle="-", linewidth=0.5)
        ax[1, 1].axhline(
            y=pct_variancia_acumulada, color="r", linestyle="-", linewidth=0.5
        )
    plt.show()
    fig.savefig(buff, format="jpeg", bbox_inches="tight")
    st.image(buff)


@st.cache_data
def calcular_ausentes_por_coluna(df):
    ausentes_por_coluna = df.isna().sum()
    return ausentes_por_coluna


def transformacao_padrao(df):
    """
    Função para padronizar os dados de um DataFrame.
    """
    scaler = StandardScaler()
    df_padrao = scaler.fit_transform(df)
    return df_padrao


def ajuste_modelo_pca(df1_padrao):
    """
    Função para ajustar o modelo PCA aos dados padronizados.
    """
    pca = PCA(n_components=18)
    pca.fit(df1_padrao)
    return pca


def preprocessamento(df):
    # Dicionário para mapear nomes dos meses (em inglês) para seus respectivos números
    numeracao_meses = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "June": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    df["Month"] = df["Month"].map(numeracao_meses)
    df = pd.get_dummies(df, columns=["VisitorType"], drop_first=True)
    return df


# Funçoes para avaliação de performance dos agrupamentos em relação as variáveis explicativas
# ---
# plotagem da proporção de cada categoria de uma coluna do DataFrame
def plotar_proporcao(coluna: str, df, posicao=st):
    # Calcula a proporção (%) de cada grupo na coluna selecionada
    fig, ax = plt.subplots(figsize=(14, 8), sharex=True)
    proporcao = df[coluna].value_counts(normalize=True) * 100
    # Plota um gráfico de barras das proporções
    proporcao.plot(kind="bar", color="skyblue")

    # Adiciona o valor percentual acima de cada barra do gráfico
    for i, valor in enumerate(proporcao):
        plt.text(i, valor + 0.5, f"{valor:.2f}%", ha="center", va="bottom", fontsize=10)

    plt.title(f"Proporção dos grupos em {coluna}")  # Define o título do gráfico
    plt.xlabel("Grupos")  # Rótulo do eixo X
    plt.ylabel("Proporção (%)")  # Rótulo do eixo Y
    plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo X para melhor visualização
    plt.show()  # Exibe o gráfico
    plt.tight_layout()
    fig.savefig(buff, format="jpeg", bbox_inches="tight")
    posicao.image(buff)


def relacao_variaveis_x(grupo: int, df, lista, posicao=st):
    # Plota gráfico de barras mostrando a média da variável 'coluna' por cluster e por Revenue
    fig, ax = plt.subplots(figsize=(14, 8), sharex=True)
    coluna = st.selectbox(
        "Escolha uma variável para realizar a comparação entre os grupos na terceira imagem",
        lista,
    )
    ax = sns.barplot(
        data=df,
        x="Grupo_" + str(grupo),
        y=coluna,
        hue="Revenue",
        ci=None,
        palette="coolwarm",
    )

    # Adiciona os valores no topo de cada barra
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=10)

    plt.title(f"Média de '{coluna}' por Cluster e Revenue")  # Título do gráfico
    plt.ylabel(f"Média de {coluna}")  # Rótulo do eixo Y
    plt.xlabel(f"Cluster (Grupo_{grupo})")  # Rótulo do eixo X
    plt.xticks(rotation=0)  # Mantém os rótulos do eixo X na horizontal
    plt.tight_layout()  # Ajusta o layout para não sobrepor elementos
    plt.show()  # Exibe o gráfico
    fig.savefig(buff, format="jpeg", bbox_inches="tight")
    posicao.image(buff)


buff = io.BytesIO()

# Executa a função main para iniciar o aplicativo Streamlit
main()

# Cria a barra lateral do aplicativo Streamlit
# Gera um campo de upload de arquivo na barra lateral para o usuário carregar um arquivo CSV
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")

# Define o caminho do arquivo da imagem que será exibida no aplicativo
imagem_rotulo_app = "Varig_logo.png"

st.markdown("---")

# Cria uma divisão de layout em três colunas, sendo a coluna do meio (col2) duas vezes maior que as laterais
col1, col2, col3 = st.columns([1, 2, 1])

# Exibe a imagem na coluna do meio (col2) com largura de 200 pixels e legenda personalizada
col2.image(
    imagem_rotulo_app, use_container_width=True, caption="Varig - Agrupamento de Dados"
)

col1, col2, col3 = st.columns([1, 3, 1])

col2.markdown("## Tratamento, limpeza e conversão para PCA do conjunto de dados")

# Verifica se um arquivo foi carregado pelo usuário na barra lateral
if uploaded_file is not None:
    # Lê o arquivo CSV carregado e armazena em um DataFrame do pandas
    df = pd.read_csv(uploaded_file)

    # Verifica a quantidade de valores ausentes em cada coluna do DataFrame

    col1, col2, col3 = st.columns([1, 1, 1])

    # Exibe a quantidade de valores ausentes por coluna no aplicativo Streamlit
    if st.checkbox("Exibir valores ausentes por coluna"):
        col1, col2, col3 = st.columns([1, 1, 1])
        # Exibe a quantidade de valores ausentes por coluna no aplicativo Streamlit
        col1.markdown(
            """
        <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

        <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 1.2em; align-items: center;'>
            <strong>Quantidade de valores ausentes por coluna</strong>
        </h1>
        """,
            unsafe_allow_html=True,
        )
        col1.dataframe(calcular_ausentes_por_coluna(df))

    # Cria uma cópia do DataFrame original para evitar alterações indesejadas no df original
    df1 = df.copy()

    # Converte a coluna 'Month' de nomes dos meses para números inteiros usando o dicionário de mapeamento
    # e aplica o pré-processamento para transformar a coluna 'VisitorType' em variáveis dummy
    # usando a função preprocessamento definida acima
    df1 = df1.pipe(preprocessamento)

    # Padroniza os dados do DataFrame usando a função transformacao_padrao
    # que aplica a normalização z-score (média 0 e desvio padrão 1)
    df1_padrao = transformacao_padrao(df1)

    # Ajusta o modelo PCA para 18 componentes principais usando os dados padronizados
    prcomp = ajuste_modelo_pca(df1_padrao)

    # Aplica a transformação PCA aos dados normalizados, reduzindo a dimensionalidade
    df_pca = prcomp.transform(df1_padrao)

    # Exibe o scree plot e as informações sobre a variância explicada
    if st.checkbox("Demonstrar dados da otimização PCA"):
        info = calcular_info_screeplot(
            prcomp.explained_variance_,
            prcomp.explained_variance_ratio_,
            prcomp.n_components_,
            varexplicada=0.90,
        )

        st.markdown(f"Número de componentes:............... {info['ncomp_crit']}")
        st.markdown(f"Variância da ultima CP:.............. {info['variancia']:.2f}")
        st.markdown(
            f"Variância total explicada:........... {info['variancia_acumulada']:.2f}"
        )
        st.markdown(
            f"Variância percentual da última CP:... {100*info['pct_variancia']:.2f}%"
        )
        st.markdown(
            f"Variância percentual total explicada: {100*info['pct_variancia_acumulada']:.2f}%"
        )

        st.session_state["calcular_info_screeplot"] = info

    # Plota os gráficos de variância explicada e percentual explicada
    if st.checkbox("Faça os gráficos para visualização"):
        plotagem_de_graficos_silhueta(
            prcomp.explained_variance_,
            prcomp.explained_variance_ratio_,
            prcomp.n_components_,
            ncomp_crit=st.session_state["calcular_info_screeplot"]["ncomp_crit"],
        )

    # Checkbox para autorizar o inicio de um processo de testes com múltiplos modelos de agrupamentos
    # Para ao final termos um conjunto de dados com variáveis que indicam os agrupamentos com diferentes números de grupos

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    col2.markdown("## Análise de diferentes modelos de agrupamento")

    # Amostra os dados se forem grandes
    if len(df_pca) > 1000:
        df_amostra = resample(df_pca, n_samples=1000, random_state=42)
    else:
        df_amostra = df_pca.copy()

    if st.checkbox(
        "Faça a avaliação de múltiplos modelos de agrupamento, para definir qual o melhor sucedido"
    ):

        df_pca = pd.DataFrame(
            df_pca, columns=[f"PC{i+1}" for i in range(18)]
        )  # Cria um DataFrame com os componentes principais

        # Lista para armazenar os scores de silhueta de cada solução de agrupamento
        silhuette_scores_pca = []

        # Número máximo de clusters a ser testado
        max_clusters = st.number_input(
            "Defina o número máximo de clusters",
            help="Decida a quantidade de clusters que serão testados. Quanto maior a quantidade, maior será o processamento",
            min_value=2,
            max_value=20,
            step=1,
            value=3,
        )

        # Dicionário para armazenar os rótulos dos grupos como categorias nomeadas
        dict_grupos_pca = {}

        # Loop para testar diferentes quantidades de clusters (de 2 até max_clusters)
        for n_clusters in range(2, max_clusters + 1):
            # Instancia o modelo KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            # Ajusta o modelo aos dados transformados pelo PCA
            kmeans.fit(df_pca)
            # Calcula e armazena o índice de silhueta
            # silhuette_scores_pca.append(silhouette_score(df_pca, kmeans.labels_))
            silhouette_score(df_amostra, kmeans.predict(df_amostra))
            # Cria nomes para os grupos (ex: Grupo_0, Grupo_1, ...)
            nomes_grupos = [f"Grupo_{i}" for i in range(n_clusters)]
            # Adiciona ao DataFrame os rótulos dos grupos (como números)
            df_pca[f"Grupo_{n_clusters}"] = kmeans.labels_
            # Adiciona ao dicionário os rótulos dos grupos como categorias nomeadas (ex: 'Grupo_0', 'Grupo_1', ...)
            dict_grupos_pca[f"Grupo_{n_clusters}"] = pd.Categorical.from_codes(
                kmeans.labels_, categories=nomes_grupos
            )

        # loop for para atribuir nomes editados a cada um dos clusters nas colunas
        # nome_coluna carrega as keys do dicionário
        # valores carrega as listas atribuidas a cada chave
        for nome_coluna, valores in dict_grupos_pca.items():
            df_pca[nome_coluna] = valores

        colunas_grupos = [col for col in df_pca.columns if col.startswith("Grupo_")]

        df1 = df1.reset_index().merge(df_pca[colunas_grupos].reset_index(), on="index")
        df1.drop(columns=["index"], inplace=True)
        st.dataframe(df1[colunas_grupos])

        # Cria um DataFrame com o número de clusters testados e os respectivos valores médios do índice de silhueta
        df_silueta_pca = pd.DataFrame(
            {
                "n_clusters": list(range(2, max_clusters + 1)),
                "silueta_média": silhuette_scores_pca,
            }
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(
            data=df_silueta_pca, x="n_clusters", y="silueta_média", markers="o"
        )
        plt.tight_layout()
        fig.savefig(buff, format="jpeg", bbox_inches="tight")
        col1, col2, col3 = st.columns([1, 1, 1])
        col1.markdown(
            """
        <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

        <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 1.2em; align-items: center;'>
            <strong>Score das silhuetas</strong>
        </h1>
        """,
            unsafe_allow_html=True,
        )
        col1.image(buff)

        col2.markdown(
            """
        <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

        <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 1.2em; align-items: center;'>
            <strong>Distribuição dos grupos</strong>
        </h1>
        """,
            unsafe_allow_html=True,
        )
        plotar_proporcao(coluna="Grupo_" + str(max_clusters), df=df_pca, posicao=col2)
        col3.markdown(
            """
        <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

        <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 1.2em; align-items: center;'>
            <strong>Proporção dos grupos para a variável escolhida</strong>
        </h1>
        """,
            unsafe_allow_html=True,
        )
        relacao_variaveis_x(
            grupo=max_clusters, df=df1, lista=df1.columns.to_list(), posicao=col3
        )
