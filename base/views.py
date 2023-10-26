from django.shortcuts import render, redirect
from django.views import View
from .forms import FileUploadForm
import pandas as pd
from .utils import *


def index(request):
	return HttpResponse('<h1>index page</h1>')

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
		feature_names = ['Yp', 'Ys', 'TOLL', 'MP', 'GMP', 'HM', 'SSI', 'STI', 'YI', 'YSI', 'RSI']
		indices_df = get_indices_df(genotype_code, Yp, Ys, RC, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI)
		ranks_df = get_ranks_df(genotype_code, Yp, Ys, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI)
		correlations_heatmaps = generate_correlations_heatmaps_images(indices_df)
		frequencies = [generate_relative_frequency_bar_graph_image(indices_df, feature_name) for feature_name in feature_names]
		frequencies = zip(feature_names, frequencies)
		context = {
			'indices_df': indices_df.to_html(),
			'ranks_df': ranks_df.to_html(),
			'3dplot': px.scatter_3d(indices_df, x='Yp', y='Ys', z='TOLL', color=indices_df.columns[0]).to_html(),
			'correlations_heatmaps': correlations_heatmaps,
			'frequencies': frequencies,

		}
		return render(self.request, 'base/results.html', context=context)
		return redirect('base:index')