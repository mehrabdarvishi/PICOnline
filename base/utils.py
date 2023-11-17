import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import math
import base64
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bioinfokit.visuz import cluster


plt.switch_backend('agg')

def get_indices_df(genotype_code, Yp, Ys, RC, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI, SI, ATI, SSPI, REI, K1STI, K2STI, SDI, DI, RDI, SNPI):
    return pd.DataFrame({        
                    'Genotype Code': genotype_code,
                    'Yp':Yp,
                    'Ys':Ys,
                    'RC':RC,
                    'TOLL':TOLL,
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
                    'RDI':RDI,
                    'SNPI':SNPI,
            })

def get_ranks_df(genotype_code, Yp, Ys, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI, SI, ATI, SSPI, REI, K1STI, K2STI, SDI, DI, RDI, SNPI, CSI):
    Yp_ranks = Yp.rank(ascending=False, method='min').astype(int)
    Ys_ranks = Ys.rank(ascending=False, method='min').astype(int)
    TOLL_ranks = TOLL.rank(ascending=True, method='min').astype(int)
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
    RDI_ranks = RDI.rank(ascending=False, method='min').astype(int)
    SNPI_ranks = SNPI.rank(ascending=False, method='min').astype(int)
    CSI_ranks = CSI.rank(ascending=False, method='min').astype(int)

    ranks_df = pd.DataFrame({
                            'Genotype Code': genotype_code,
                            'Yp':Yp_ranks,
                            'Ys':Ys_ranks,
                            'TOLL':TOLL_ranks,
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
                            'RDI':RDI_ranks,
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
    ax.hist(df[feature_name], edgecolor='black', weights=np.ones_like(df[feature_name]) / number_of_rows, bins=7)
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
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig





def generate_bar_chart_2(df, feature_name):
    fig = plt.figure(figsize = (20, 10))
    plt.bar(df.iloc[:, 0], df[feature_name], color ='#24f3b9', align='center')
    plt.xlabel(feature_name, fontsize=25, labelpad=15)
    plt.ylabel('value', fontsize=25, labelpad=15)
    #plt.title(f'Bar chart', fontsize=32)
    plt.xticks(rotation=90)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close(fig)
    return graph



def generate_correlations_heatmaps_images(df):
    fig, axs = plt.subplots(2, 1, figsize=(25, 25))
    df = df.drop([df.columns[0], 'RC'], axis=1)
    Pearsons_heatmap = df.corr(method='pearson')

    spearmans_heatmap = df.corr(method='spearman')
    plot = sns.heatmap(Pearsons_heatmap, annot=True, square=True, ax=axs[0])
    axs[0].set_title("Pearson's Correlation Heatmap")
    plot = sns.heatmap(spearmans_heatmap, annot=True, square=True, ax=axs[1])
    axs[1].set_title("Spearman's Correlation Heatmap")
    buffer = BytesIO()
    plot.get_figure().savefig(buffer, bbox_inches='tight', format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close(fig)
    return graph


'''
def generate_pca_plot_image(df):
    X = df.select_dtypes(include=np.number)
    X_st =  StandardScaler().fit_transform(X)
    pca_out = PCA().fit(X_st)
    loadings = pca_out.components_
    pca_out.explained_variance_
    pca_scores = PCA().fit_transform(X_st)
    c = cluster.biplot(cscore=pca_scores, loadings=loadings, labels=X.columns.values, var1=round(pca_out.explained_variance_ratio_[0]*100, 2),
        var2=round(pca_out.explained_variance_ratio_[1]*100, 2))
    print(help(cluster))
'''


def generate_pca_plot_image(df):
    X = df.select_dtypes(include=np.number)
    labels = X.columns.values
    scaler = StandardScaler()
    scaler.fit(X)
    X=scaler.transform(X)    
    pca = PCA()
    x_new = pca.fit_transform(X)
    # Use only the 2 PCs.
    score = x_new[:,0:2]
    coeff = np.transpose(pca.components_[0:2, :])
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    fig = plt.figure()
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close(fig)
    return graph




if __name__ == '__main__':
    df = pd.read_excel('Example#1.xlsx')
    genotype_code = df[df.columns[0]]
    Yp, Ys = df.Yp, df.Ys
    Yp_mean, Ys_mean = Yp.mean(), Ys.mean()
    RC = (Yp - Ys) / Yp * 100
    TOLL = Yp - Ys
    MP = (Yp + Ys) / 2
    GMP = np.sqrt(Ys * Yp)
    HM = 2 * Ys * Yp / (Ys + Yp)
    SSI = (1 - Ys / Yp) / (1 - Ys_mean / Yp_mean)
    STI = Ys * Yp / Yp_mean ** 2
    YI = Ys / Ys_mean
    YSI = Ys / Yp
    RSI = (Ys / Yp) / (Ys_mean / Yp_mean)
    indices_df = get_indices_df(genotype_code, Yp, Ys, RC, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI)
    ranks_df = get_ranks_df(genotype_code, Yp, Ys, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI)



