document.querySelectorAll(".results-container .result .tables .select-table .toggle-button").forEach(element => {
    element.addEventListener("click", (event)=>{
        
        selected_table_data_attr = event.target.getAttribute("data-title");
    	const other_table = document.querySelector(`.results-container .result .tables .dataframe-container:not([data-table-type=${selected_table_data_attr}])`);
        other_table.style.display = "none";
        document.querySelector(`.results-container .result .tables .dataframe-container[data-table-type=${selected_table_data_attr}]`).style.display = "block";
        element.classList.add("selected");
        document.querySelector(`.results-container .result .tables .select-table .toggle-button:not([data-title=${selected_table_data_attr}])`).classList.remove("selected");
    });
});

const indices_table = document.querySelector('.results-container .result .tables .dataframe-container[data-table-type=indices] table');
const indices_table_download_icon = document.querySelector('.results-container .result .tables .indices-table-download-icon');
indices_table_download_icon.addEventListener('click', (event)=>{
    var event = new Event('click');
    document.querySelector('.results-container .result .tables .select-table .toggle-button[data-title=indices]').dispatchEvent(event);
    const table2excel = new Table2Excel();
    table2excel.export(indices_table, 'indices');
});


const ranks_table = document.querySelector('.results-container .result .tables .dataframe-container[data-table-type=ranks] table');
const ranks_table_download_icon = document.querySelector('.results-container .result .tables .ranks-table-download-icon');
ranks_table_download_icon.addEventListener('click', (event)=>{
    var event = new Event('click');
    document.querySelector('.results-container .result .tables .select-table .toggle-button[data-title=ranks]').dispatchEvent(event);
    const table2excel = new Table2Excel();
    table2excel.export(ranks_table, 'ranks');
})


const describe_table = document.querySelector('.results-container .result .describe table');
const describe_table_download_icon = document.querySelector('.results-container .result .describe .describe-table-download-icon');
describe_table_download_icon.addEventListener('click', (event)=>{
    const table2excel = new Table2Excel();
    table2excel.export(describe_table, 'describe');
})
