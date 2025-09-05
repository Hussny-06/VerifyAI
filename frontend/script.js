// script.js (Corrected Version)

document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const videoUpload = document.getElementById('videoUpload');
    const imagePreview = document.getElementById('imagePreview');
    const videoPreview = document.getElementById('videoPreview');
    const analyzeButton = document.getElementById('analyzeButton');
    const analyzeVideoButton = document.getElementById('analyzeVideoButton');
    const predictionResult = document.getElementById('prediction');
    const confidenceLevel = document.getElementById('confidenceLevel');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const imageUploadArea = document.getElementById('image-upload-area');
    const videoUploadArea = document.getElementById('video-upload-area');

    let selectedImage = null;
    let selectedVideo = null;

    // Drag and drop functionality is great! No changes needed here.
    function setupDragDrop(area, input) {
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('drag-over');
        });
        area.addEventListener('dragleave', () => {
            area.classList.remove('drag-over');
        });
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('drag-over');
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                const event = new Event('change');
                input.dispatchEvent(event);
            }
        });
    }

    setupDragDrop(imageUploadArea, imageUpload);
    setupDragDrop(videoUploadArea, videoUpload);

    // Image upload handling is also good. No changes needed.
    imageUpload.addEventListener('change', (event) => {
        selectedImage = event.target.files[0];
        if (selectedImage) {
            imagePreview.innerHTML = '';
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.classList.add('preview-media');
                imagePreview.appendChild(img);
            };
            reader.readAsDataURL(selectedImage);
            analyzeButton.disabled = false;
        }
    });

    // Video upload handling is also fine.
    videoUpload.addEventListener('change', (event) => {
        selectedVideo = event.target.files[0];
        if (selectedVideo) {
            videoPreview.innerHTML = '';
            const video = document.createElement('video');
            video.src = URL.createObjectURL(selectedVideo);
            video.controls = true;
            video.classList.add('preview-media');
            videoPreview.appendChild(video);
            analyzeVideoButton.disabled = false;
        }
    });

    // This is the main function to fix.
    async function sendToBackend(file, type) {
        // --- FIX #2: Change the FormData key to 'image' ---
        const formData = new FormData();
        formData.append('image', file); // The backend expects the key to be 'image'

        predictionResult.textContent = `Analyzing ${type}...`;
        confidenceLevel.style.width = '0%';
        progressContainer.style.display = 'block';
        progressBar.style.width = '30%';

        try {
            // --- FIX #1: Change the URL to '/analyze' ---
            const response = await fetch('/analyze', { // The backend route is '/analyze'
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Server error: " + response.statusText);

            const result = await response.json();
            progressBar.style.width = '100%';
            document.getElementById('result-section-id').scrollIntoView({ behavior: 'smooth', block: 'start' });

            // --- FIX #3: Handle the actual JSON response from the backend ---
            // The backend sends 'prediction' and 'confidence' (as a string like "95.24%")
            const confidenceValue = parseFloat(result.confidence); // Convert "95.24%" to 95.24
            
            predictionResult.innerHTML = `${result.prediction}  
Confidence: ${result.confidence}`;
            confidenceLevel.style.width = `${confidenceValue}%`; // Use the number for the width

        } catch (error) {
            predictionResult.textContent = "Error analyzing file. Please try again.";
            console.error(error);
        } finally {
            // A small improvement: hide the progress bar after a short delay
            setTimeout(() => {
                progressContainer.style.display = 'none';
            }, 1000);
        }
    }

    analyzeButton.addEventListener('click', () => {
        if (selectedImage) sendToBackend(selectedImage, 'image');
    });

    analyzeVideoButton.addEventListener('click', () => {
        // Note: The current backend only supports images. This will fail.
        // For now, we can show a message.
        alert("Video analysis is not yet implemented in the backend.");
        // if (selectedVideo) sendToBackend(selectedVideo, 'video');
    });
});
