const select_feature = document.querySelector('.results-container .result .frequencies .select-feature select');
select_feature.addEventListener("change", event => {
	selected_feature_graph = document.querySelector(`.results-container .result .frequencies img[alt="${event.target.value}"]`);
	document.querySelectorAll(`.results-container .result .frequencies img:not([alt="${event.target.value}"])`).forEach(element => {
		element.style.display = 'none'
	});
	selected_feature_graph.style.display = 'block';
});