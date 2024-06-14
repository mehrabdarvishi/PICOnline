const select_barchart_feature = document.querySelector('.results-container .result .bar-charts .select-feature select');
select_barchart_feature.addEventListener("change", event => {
	selected_feature_graph = document.querySelector(`.results-container .result .bar-charts .bar-chart[data-feature-name="${event.target.value}"]`);
	document.querySelectorAll(`.results-container .result .bar-charts .bar-chart:not([data-feature-name="${event.target.value}"])`).forEach(element => {
		element.style.display = 'none'
	});
	selected_feature_graph.style.display = 'flex';
	selected_feature_graph.style.justifyContent = "center";
});