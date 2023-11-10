var three_d_plot_mode = 'colored'
const three_d_plot_switch_mode_btn = document.querySelector('.results-container .result .three-d-plot .switch_plot_btn');
three_d_plot_switch_mode_btn.addEventListener('click', event => {
	black_and_white_plot = document.querySelector('.results-container .result .three-d-plot .black-and-white');
	colored_plot = document.querySelector('.results-container .result .three-d-plot .colored');
	if (three_d_plot_mode == 'colored') {
		black_and_white_plot.style.display = 'block';
		colored_plot.style.display = 'none';
		three_d_plot_mode = 'black-and-white';
		three_d_plot_switch_mode_btn.style.backgroundColor = '#6cebb4';
		three_d_plot_switch_mode_btn.style.color = '#000';
		three_d_plot_switch_mode_btn.innerHTML = '<span>Colored mode</span>'
	}else {
		black_and_white_plot.style.display = 'none';
		colored_plot.style.display = 'block';
		three_d_plot_mode = 'colored';
		three_d_plot_switch_mode_btn.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
		three_d_plot_switch_mode_btn.style.color = '#f1f1f1';
		three_d_plot_switch_mode_btn.innerHTML = '<span>Black and white mode</span>'
	}
});