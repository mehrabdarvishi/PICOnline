const select_pc1 = document.getElementById('pc1-select');
const select_pc2 = document.getElementById('pc2-select');
document.querySelectorAll(".results-container .result .pca .plotly-graph-div:not(#PC1-PC2").forEach(element => {
    element.style.display = 'none';
});
select_pc1.addEventListener("change", event => {
    selected_figure_id = `${event.target.value}-${select_pc2.value}`
	selected_figure = document.getElementById(selected_figure_id);
    console.log(selected_figure)
    
    
	document.querySelectorAll(`.results-container .result .pca .plotly-graph-div:not(#${selected_figure_id}`).forEach(element => {
		element.style.display = 'none'
	});
	selected_figure.style.display = 'flex';
	selected_figure.style.justifyContent = "center";
});
select_pc2.addEventListener("change", event => {
    selected_figure_id = `${select_pc1.value}-${event.target.value}`
	selected_figure = document.getElementById(selected_figure_id);
    console.log(selected_figure)
    
    
	document.querySelectorAll(`.results-container .result .pca .plotly-graph-div:not(#${selected_figure_id}`).forEach(element => {
		element.style.display = 'none'
	});
	selected_figure.style.display = 'flex';
	selected_figure.style.justifyContent = "center";
});