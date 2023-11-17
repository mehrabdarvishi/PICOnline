const select_barchart_feature = document.querySelector('.results-container .result .bar-charts .select-feature select');
select_barchart_feature.addEventListener("change", event => {
	selected_feature_graph = document.querySelector(`.results-container .result .bar-charts img[alt="${event.target.value}"]`);
	document.querySelectorAll(`.results-container .result .bar-charts img:not([alt="${event.target.value}"])`).forEach(element => {
		element.style.display = 'none'
	});
	selected_feature_graph.style.display = 'block';
});