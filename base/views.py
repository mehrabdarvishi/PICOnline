from django.shortcuts import render, HttpResponse, get_object_or_404
from django.views import View 
from .forms import FileUploadForm
import pandas as pd
from .utils import *
from . import models


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
		TOL = Yp - Ys
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
		SNPI = np.cbrt(((Yp + Ys) / (Yp - Ys))) * np.cbrt(Yp * Ys * Ys)
		indices_df = get_indices_df(genotype_code, Yp, Ys, TOL, MP, GMP, HM, SSI, STI, YI, YSI, RSI, SI, ATI, SSPI, REI, K1STI, K2STI, SDI, DI, SNPI)
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
		ranks_df = get_ranks_df(genotype_code, Yp, Ys, TOL, MP, GMP, HM, SSI, STI, YI, YSI, RSI, SI, ATI, SSPI, REI, K1STI, K2STI, SDI, DI, SNPI, CSI)
		indices_df['CSI'] = CSI

		rsp = HttpResponse(content_type='application/xlsx')
		rsp['Content-Disposition'] = f'attachment; filename="indices.xlsx"'
		with pd.ExcelWriter(rsp) as writer:
			df.to_excel(writer, sheet_name='SHEET NAME')

		feature_names = ['Yp', 'Ys', 'TOL', 'MP', 'GMP', 'HM', 'SSI', 'STI', 'YI', 'YSI', 'RSI', 'SI', 'ATI', 'SSPI', 'REI', 'K1STI', 'K2STI', 'SDI', 'DI', 'SNPI', 'CSI']

		#pearson_heatmap = generate_correlations_heatmap_image(indices_df, method='pearson')
		#spearman_heatmap = generate_correlations_heatmap_image(indices_df, method='spearman')
		pearson_heatmap = generate_correlations_heatmap(indices_df, method='pearson').to_html(full_html=False)
		spearman_heatmap = generate_correlations_heatmap(indices_df, method='spearman').to_html(full_html=False)
		pca_data = generate_pca_data(indices_df)
		pca_plots = pca_data['figures']
		number_of_pcs = pca_data['number_of_pcs']
		bar_charts = zip(feature_names, [generate_bar_chart(indices_df, feature_name).to_html(full_html=False, include_plotlyjs=True if index == 0 else False, include_mathjax=False) for index, feature_name in enumerate(feature_names)])
		frequencies = [generate_relative_frequency_bar_graph_image(indices_df, feature_name) for feature_name in feature_names]
		frequencies = zip(feature_names, frequencies)
		describe = indices_df.describe()
		describe.loc['median'] = numeric_indices.median()
		describe.loc['variance'] = numeric_indices.var()
		context = {
			'indices_df': indices_df.to_html(),
			'ranks_df': ranks_df.to_html(),
			'describe': describe.to_html(),
			'3dplot': generate_3d_plot(indices_df, x='Yp', y='Ys', z='TOL').to_html(full_html=False, include_plotlyjs=False, include_mathjax=False),
			'3dplot_black_and_white': generate_3d_plot(indices_df, x='Yp', y='Ys', z='TOL', black_and_white=True).to_html(full_html=False, include_plotlyjs=False, include_mathjax=False),
			'pearson_heatmap': pearson_heatmap,
			'spearman_heatmap': spearman_heatmap,
			'bar_charts': bar_charts,
			'frequencies': frequencies,
			'pca_plots': pca_plots,
			'number_of_pcs': range(1, number_of_pcs + 1),
			'rsp': rsp,
		}
		return render(self.request, 'base/results.html', context=context)




class MenuView(View):
	def get(self, *args, **kwargs):
		url=kwargs['slug']
		menu=get_object_or_404(models.Menu,url=url,active=True)
		context = {
			'body':menu.body,
		}
		return render(self.request, 'base/custopm_page.html', context=context)

class ContactView(View):
	def get(self, *args, **kwargs):
		return render(self.request, 'base/contact.html', {})
	
	def post(self, request ,*args, **kwargs):
		message=None
		new_contact=models.Contact.objects.create(
			fullname=request.POST.get('fullname'),
			email=request.POST.get('email'),
			message=request.POST.get('message')
		)
		new_contact.save()
		message='Your message has been sent.'

		context = {
			'message':message,	
		}
		
		return render(self.request, 'base/contact.html', context=context)