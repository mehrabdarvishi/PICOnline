document.getElementById('id_file').addEventListener('change', function(e) {
  if (e.target.files[0]) {
  	const label = document.querySelector('.form-container label');
  	label.style.fontSize = '1.2rem';
  	label.innerHTML = e.target.files[0].name.slice(-20);
  	const generate_btn = document.querySelector('.form-container button');
  	generate_btn.style.display = 'block';
  }
});
