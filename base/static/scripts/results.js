document.querySelectorAll(".results-container .results-bar ul li").forEach(element => {
    element.addEventListener("click", event => {
        document.querySelectorAll(".results-container .results-bar ul li").forEach(element => {
            element.classList.remove("selected");
        });
        element.classList.add("selected");
    });
});