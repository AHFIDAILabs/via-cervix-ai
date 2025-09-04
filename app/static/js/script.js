const imageUpload = document.getElementById('image-upload');
const predictButton = document.getElementById('predict-button');
const resultDiv = document.getElementById('result');

predictButton.addEventListener('click', async () => {
    if (!imageUpload.files[0]) {
        resultDiv.innerHTML = 'Please select an image first.';
        return;
    }

    const formData = new FormData();
    formData.append('file', imageUpload.files[0]);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
    });

    const data = await response.json();
    resultDiv.innerHTML = `Prediction: ${data.prediction} (Confidence: ${data.confidence})`;
});