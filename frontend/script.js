document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeButton = document.getElementById('analyzeButton');
    const predictionResult = document.getElementById('prediction');
    const uploadLabel = document.querySelector('.upload-label span');

    let selectedFile = null;

    imageUpload.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            // Clear previous preview
            imagePreview.innerHTML = '';
            
            // Create and display new preview
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                imagePreview.appendChild(img);
            };
            reader.readAsDataURL(selectedFile);

            uploadLabel.textContent = selectedFile.name; // Show file name
            analyzeButton.disabled = false; // Enable the button
        }
    });

    // In frontend/script.js

analyzeButton.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first.');
        return;
    }

    predictionResult.textContent = 'Analysis in progress...';
    analyzeButton.disabled = true; // Disable button during analysis

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        // Use the 'fetch' API to send the image to our Flask backend
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData, // The FormData object is the body of the request
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // Get the JSON result from the backend
        const result = await response.json();

        // Display the result from the backend
        predictionResult.textContent = `Prediction: ${result.prediction} (Confidence: ${result.confidence})`;

    } catch (error) {
        console.error('Error during analysis:', error);
        predictionResult.textContent = 'Analysis failed. Please try again.';
    } finally {
        analyzeButton.disabled = false; // Re-enable the button
    }
});

});
