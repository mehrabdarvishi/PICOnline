document.querySelectorAll(".results-container .result .tables .select-table span").forEach(element => {
    element.addEventListener("click", (event)=>{
        
        selected_table_data_attr = event.target.getAttribute("data-title");
    	const other_table = document.querySelector(`.results-container .result .tables .dataframe-container:not([data-table-type=${selected_table_data_attr}])`);
        other_table.style.display = "none";
        document.querySelector(`[data-table-type=${selected_table_data_attr}]`).style.display = "block";
        element.classList.add("selected");
        document.querySelector(`.results-container .result .tables .select-table span:not([data-title=${selected_table_data_attr}])`).classList.remove("selected");
    });
});