document.querySelectorAll(".results-bar ul li").forEach(element => {
    element.addEventListener("click", (event)=>{
        
        selected_tab_data_attr = event.target.getAttribute("data-result-name");
    	const other_results = document.querySelectorAll(`.results-container .result > div:not(.${selected_tab_data_attr})`);
    	other_results.forEach(other_result => {
            other_result.style.display = "none";
        });
        document.getElementsByClassName(selected_tab_data_attr)[0].style.display = "block";        
    });
});