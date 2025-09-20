// Image upload functionality
document.getElementById('uploadButton').addEventListener('click', function () {
    const input = document.createElement('input');
    input.type = 'file';  
    input.accept = 'image/*';  
    input.onchange = function (event) {
      const file = event.target.files[0];  
      if (file) {
        const reader = new FileReader(); 
        reader.onload = function () {
          document.getElementById('uploadedImage').src = reader.result;
  
          saveImage(reader.result);
        };
        reader.readAsDataURL(file);  
      }
    };
    input.click(); 
  });
  
  function saveImage(imageData) {
    console.log("Uploaded Image Data:", imageData);
  
    localStorage.setItem('capturedImage', imageData);
    alert('Image uploaded and saved for analysis!');
  }

  window.onload = function() {
    const capturedImage = localStorage.getItem('capturedImage');  
    if (capturedImage) {
      document.getElementById('uploadedImage').src = capturedImage;  
    }
  };
  
  document.getElementById('forms').addEventListener('submit', function(e) {
    e.preventDefault(); 

    const selectedAge = document.querySelector('input[name="age"]:checked');
    const selectedSkinType = document.querySelector('input[name="skin-type"]:checked');

    if (!selectedAge || !selectedSkinType || !document.getElementById('firstName').value || !document.getElementById('lastName').value) {
        alert("Please fill in all the fields.");
        return;
    }

    document.querySelector('.title').textContent = "We are processing your skin analysis...";
    document.querySelector('.smallContainer p').textContent = "Please wait a moment while we analyze the data you've provided.";

    setTimeout(function() {
        document.querySelector('.title').textContent = "Analysis Complete!";
        document.querySelector('.smallContainer p').textContent = "We have analyzed your skin. See the result below!";

        document.querySelector('.smallContainer form').innerHTML = "<h2>Analysis Result:</h2><p>Skin Type: " + selectedSkinType.value + "</p><p>Age Group: " + selectedAge.value + "</p>"; // Example result
    }, 3000); 
  });
