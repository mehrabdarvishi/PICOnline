from django.shortcuts import render, redirect
from django.views import View
from .forms import FileUploadForm
import pandas as pd
from .utils import *



class IndexView(View):
	def get(self, *args, **kwargs):
		context = {
			'form': FileUploadForm()
		}
		return render(self.request, 'base/index.html', context=context)

	def post(self, *args, **kwargs):
		uploaded_file = self.request.FILES['file']

		df = pd.read_excel(uploaded_file)
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
		SI = 1 - (Ys / Yp)
		ATI = ((Yp - Ys) / (Yp_mean/Ys_mean)) * np.sqrt(Ys * Yp)
		SSPI = ((Yp - Ys) / 2 * Yp_mean) * 100
		REI = (Ys / Ys_mean) * (Yp / Yp_mean)
		K1STI = Yp * Yp * Yp_mean * Yp_mean
		K2STI = Ys * Ys * Ys_mean * Ys_mean
		SDI = (Yp - Ys) / Yp
		DI = ((Ys / Yp) / Ys_mean) * Ys
		RDI = ((Ys / Yp) / (Ys_mean / Yp_mean))
		SNPI = np.cbrt(((Yp + Ys) / (Yp - Ys))) * np.cbrt(Yp * Ys * Ys)
		indices_df = get_indices_df(genotype_code, Yp, Ys, RC, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI, SI, ATI, SSPI, REI, K1STI, K2STI, SDI, DI, RDI, SNPI)
		numeric_indices = indices_df.drop([indices_df.columns[0]], axis=1)
		correlation_table = numeric_indices.corr(method='pearson')
		CSI = 1 / 2 * (
				(correlation_table['Yp']['MP'] * MP) +
				(correlation_table['Yp']['GMP'] * GMP) +
				(correlation_table['Yp']['HM'] * HM) +
				(correlation_table['Yp']['STI'] * STI) +
				(correlation_table['Ys']['MP'] * MP) +
				(correlation_table['Ys']['GMP'] * GMP) +
				(correlation_table['Ys']['HM'] * HM) +
				(correlation_table['Ys']['STI'] * STI)
			)
		ranks_df = get_ranks_df(genotype_code, Yp, Ys, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI, SI, ATI, SSPI, REI, K1STI, K2STI, SDI, DI, RDI, SNPI, CSI)
		indices_df['CSI'] = CSI


		feature_names = ['Yp', 'Ys', 'TOLL', 'MP', 'GMP', 'HM', 'SSI', 'STI', 'YI', 'YSI', 'RSI', 'SI', 'ATI', 'SSPI', 'REI', 'K1STI', 'K2STI', 'SDI', 'DI', 'RDI', 'SNPI', 'CSI']


		correlations_heatmaps = generate_correlations_heatmaps_images(indices_df)
		pca = generate_pca_plot_image(indices_df)
		#bar_charts = [generate_bar_chart(indices_df, feature_name).to_html() for feature_name in feature_names]
		bar_charts = zip(feature_names, [generate_bar_chart_2(indices_df, feature_name) for feature_name in feature_names])
		frequencies = [generate_relative_frequency_bar_graph_image(indices_df, feature_name) for feature_name in feature_names]
		frequencies = zip(feature_names, frequencies)
		describe = indices_df.describe()
		describe.loc['median'] = numeric_indices.median()
		describe.loc['variance'] = numeric_indices.var()
		context = {
			'indices_df': indices_df.to_html(),
			'ranks_df': ranks_df.to_html(),
			'describe': describe.to_html(),
			'3dplot': generate_3d_plot(indices_df, x='Yp', y='Ys', z='TOLL').to_html(),
			'3dplot_black_and_white': generate_3d_plot(indices_df, x='Yp', y='Ys', z='TOLL', black_and_white=True).to_html(),
			'correlations_heatmaps': correlations_heatmaps,
			'bar_charts': bar_charts,
			'frequencies': frequencies,
			'pca': pca,
		}
		return render(self.request, 'base/results.html', context=context)
		return redirect('base:index')