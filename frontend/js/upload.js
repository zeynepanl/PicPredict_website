document.getElementById('upload-area').addEventListener('click', function() {
    document.getElementById('file-input').click();
});

document.getElementById('upload-area').addEventListener('dragover', function(event) {
    event.preventDefault();
    event.stopPropagation();
    this.style.borderColor = '#333';
});

document.getElementById('upload-area').addEventListener('dragleave', function(event) {
    event.preventDefault();
    event.stopPropagation();
    this.style.borderColor = '#ccc';
});

document.getElementById('upload-area').addEventListener('drop', function(event) {
    event.preventDefault();
    event.stopPropagation();
    this.style.borderColor = '#ccc';
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('file-input').files = files;
        displayImage(files[0]);
    }
});

document.getElementById('file-input').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        displayImage(file);
    }
});

document.getElementById('upload-button').addEventListener('click', function(event) {
    event.preventDefault(); // Formun varsayılan submit davranışını önle

    const fileInput = document.getElementById('file-input');
    if (fileInput.files.length > 0) {
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data && data.category && data.probability !== undefined) {
                document.getElementById('category').innerText = data.category;
                document.getElementById('probability').innerText = data.probability;
            } else {
                alert('Geçersiz tahmin verisi alındı. Lütfen tekrar deneyin.');
            }
        })
        .catch(error => {
            alert('Resim yüklenirken bir hata oluştu. Lütfen tekrar deneyin.');
        });
    } else {
        alert('Lütfen yüklemek için bir dosya seçin.');
    }
});

function displayImage(file) {
    const reader = new FileReader();
    reader.onload = function(event) {
        const imgElement = document.getElementById('uploaded-image');
        imgElement.src = event.target.result;
        imgElement.style.display = 'block';
        document.getElementById('upload-icon').style.display = 'none';
    }
    reader.readAsDataURL(file);
}
