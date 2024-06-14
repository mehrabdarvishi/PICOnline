import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
import base64
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from itertools import permutations

plt.switch_backend('agg')

def get_indices_df(genotype_code, Yp, Ys, TOL, MP, GMP, HM, SSI, STI, YI, YSI, RSI, SI, ATI, SSPI, REI, K1STI, K2STI, SDI, DI, SNPI):
    return pd.DataFrame({        
                    'Genotype Code': genotype_code,
                    'Yp':Yp,
                    'Ys':Ys,
                    'TOL':TOL,
                    'MP':MP,
                    'GMP':GMP,
                    'HM':HM,
                    'SSI':SSI,
                    'STI':STI,
                    'YI':YI,
                    'YSI':YSI,
                    'RSI':RSI,
                    'SI':SI,
                    'ATI':ATI,
                    'SSPI':SSPI,
                    'REI':REI,
                    'K1STI':K1STI,
                    'K2STI':K2STI,
                    'SDI':SDI,
                    'DI':DI,
                    'SNPI':SNPI,
            })

def get_ranks_df(genotype_code, Yp, Ys, TOL, MP, GMP, HM, SSI, STI, YI, YSI, RSI, SI, ATI, SSPI, REI, K1STI, K2STI, SDI, DI, SNPI, CSI):
    Yp_ranks = Yp.rank(ascending=False, method='min').astype(int)
    Ys_ranks = Ys.rank(ascending=False, method='min').astype(int)
    TOL_ranks = TOL.rank(ascending=True, method='min').astype(int)
    MP_ranks = MP.rank(ascending=False, method='min').astype(int)
    GMP_ranks = GMP.rank(ascending=False, method='min').astype(int)
    HM_ranks = HM.rank(ascending=False, method='min').astype(int)
    SSI_ranks = SSI.rank(ascending=True, method='min').astype(int)
    STI_ranks = STI.rank(ascending=False, method='min').astype(int)
    YI_ranks = YI.rank(ascending=False, method='min').astype(int)
    YSI_ranks = YSI.rank(ascending=False, method='min').astype(int)
    RSI_ranks = RSI.rank(ascending=False, method='min').astype(int)
    SI_ranks = SI.rank(ascending=False, method='min').astype(int)
    ATI_ranks = ATI.rank(ascending=False, method='min').astype(int)
    SSPI_ranks = SSPI.rank(ascending=False, method='min').astype(int)
    REI_ranks = REI.rank(ascending=False, method='min').astype(int)
    K1STI_ranks = K1STI.rank(ascending=False, method='min').astype(int)
    K2STI_ranks = K2STI.rank(ascending=False, method='min').astype(int)
    SDI_ranks = SDI.rank(ascending=False, method='min').astype(int)
    DI_ranks = DI.rank(ascending=False, method='min').astype(int)
    SNPI_ranks = SNPI.rank(ascending=False, method='min').astype(int)
    CSI_ranks = CSI.rank(ascending=False, method='min').astype(int)

    ranks_df = pd.DataFrame({
                            'Genotype Code': genotype_code,
                            'Yp':Yp_ranks,
                            'Ys':Ys_ranks,
                            'TOL':TOL_ranks,
                            'MP':MP_ranks,
                            'GMP':GMP_ranks,
                            'HM':HM_ranks,
                            'SSI':SSI_ranks,
                            'STI': STI_ranks,
                            'YI':YI_ranks,
                            'YSI':YSI_ranks,
                            'RSI':RSI_ranks,
                            'SI':SI_ranks,
                            'ATI':ATI_ranks,
                            'SSPI':SSPI_ranks,
                            'REI':REI_ranks,
                            'K1STI':K1STI_ranks,
                            'K2STI':K2STI_ranks,
                            'SDI':SDI_ranks,
                            'DI':DI_ranks,
                            'SNPI':SNPI_ranks,
                            'CSI':CSI_ranks,
                        })

    numeric_only_ranks_df = ranks_df.select_dtypes(include=np.number)
    SR = numeric_only_ranks_df.sum(axis=1)
    AR = SR / len(numeric_only_ranks_df.columns)
    SD = numeric_only_ranks_df.std(axis=1)

    ranks_df['SR'] = SR
    ranks_df['AR'] = AR
    ranks_df['SD'] = SD

    return ranks_df



def generate_3d_plot(df, x, y, z, black_and_white=False):
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=df.columns[0])
    if black_and_white:
        fig.update_layout(scene=dict(
                bgcolor='#fff',
                xaxis=dict(gridcolor='#000', backgroundcolor='#fff', color='#000'),
                yaxis=dict(gridcolor='#000', backgroundcolor='#fff', color='#000'),
                zaxis=dict(gridcolor='#000', backgroundcolor='#fff', color='#000'),
            )       
        )
        fig.update_traces(marker=dict(color='#000'))
    return fig


def compute_histogram_bins(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins


def generate_relative_frequency_bar_graph_image(df, feature_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    number_of_rows = df.shape[0]
    bin_size = math.floor(1 + 3.322 * math.log(number_of_rows, 10))
    plt.xlabel(feature_name)
    plt.ylabel('Relative Frequency')
    bins = compute_histogram_bins(data=df[feature_name], desired_bin_size=bin_size)
    x = df[feature_name]
    weights = np.ones_like(df[feature_name]) / number_of_rows
    print(x)
    print('*'*100)
    print(weights)
    ax.hist(x, weights=weights, bins=7, edgecolor='black',)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close(fig)
    return graph


def generate_bar_chart(df, feature_name):
    fig = px.bar(df, y=feature_name, x=df.columns[0], text=feature_name)
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', template='xgridoff')
    fig.update_layout(autosize=True)
    return fig



def generate_correlations_heatmap_image(df, method):
    fig, ax = plt.subplots(figsize=(15, 15))
    df = df.drop([df.columns[0]], axis=1)
    heatmap = df.corr(method=method)
    plot = sns.heatmap(heatmap, annot=True, square=True, ax=ax)
    ax.set_title(f"{method.capitalize()}'s Correlation Heatmap")
    buffer = BytesIO()
    plot.get_figure().savefig(buffer, bbox_inches='tight', format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close(fig)
    return graph


def generate_pca_data(df):
    genotype_code = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    df = pd.DataFrame(X, columns=list(df.columns.values))
    pca = PCA(0.999)
    components = pca.fit_transform(df)
    number_of_pcs = components.shape[1]
    pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(number_of_pcs)])
    pc_permutations = list(permutations(pca_df, 2))
    pc_permutations += [(pc, pc) for pc in pca_df.columns]
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    figures = []
    for pc1, pc2 in pc_permutations:
        pc1_number, pc2_number = int(pc1[2:])-1, int(pc2[2:])-1
        explained_variance = [pca.explained_variance_[pc1_number], pca.explained_variance_[pc2_number]]
        loadings = pca.components_[[pc1_number, pc2_number]].T * np.sqrt(explained_variance)
        fig = px.scatter(pca_df, x=pc1, y=pc2, hover_name=genotype_code, template='simple_white')
        # title=f'{pc1} vs {pc2}',
        for i, feature in enumerate(df.columns.values):
            fig.add_annotation(
                ax=0, ay=0,
                axref="x", ayref="y",
                x=loadings[i, 0],
                y=loadings[i, 1],
                showarrow=True,
                arrowsize=2,
                arrowhead=2,
                xanchor="right",
                yanchor="top"
            )
            fig.add_annotation(
                x=loadings[i, 0],
                y=loadings[i, 1],
                ax=0, ay=0,
                xanchor="center",
                yanchor="bottom",
                text=feature,
                yshift=5,
            )
        figures.append(fig.to_html(full_html=False, div_id=f'{pc1}-{pc2}', include_plotlyjs=False, include_mathjax=False))
    return {
        'figures': figures,
        'number_of_pcs': number_of_pcs
    }



if __name__ == '__main__':
    df = pd.read_excel('Example#1.xlsx')
    genotype_code = df[df.columns[0]]
    Yp, Ys = df.Yp, df.Ys
    Yp_mean, Ys_mean = Yp.mean(), Ys.mean()
    TOL = Yp - Ys
    MP = (Yp + Ys) / 2
    GMP = np.sqrt(Ys * Yp)
    HM = 2 * Ys * Yp / (Ys + Yp)
    SSI = (1 - Ys / Yp) / (1 - Ys_mean / Yp_mean)
    STI = Ys * Yp / Yp_mean ** 2
    YI = Ys / Ys_mean
    YSI = Ys / Yp
    RSI = (Ys / Yp) / (Ys_mean / Yp_mean)
    indices_df = get_indices_df(genotype_code, Yp, Ys, TOL, MP, GMP, HM, SSI, STI, YI, YSI, RSI)
    ranks_df = get_ranks_df(genotype_code, Yp, Ys, TOL, MP, GMP, HM, SSI, STI, YI, YSI, RSI)
