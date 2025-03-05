// Hugging Face Integration and Image Processing
const dropzone = document.getElementById('dropzone');
const imageUpload = document.getElementById('imageUpload');
const processingSection = document.getElementById('processing-section');
const notesContainer = document.getElementById('notes-container');
const plainTextBtn = document.getElementById('plainTextBtn');

// Hugging Face Inference API Endpoints
const INFERENCE_API_URL = 'https://api-inference.huggingface.co/models';

// Object Detection Model (Public)
const OBJECT_DETECTION_MODEL = 'facebook/detr-resnet-50';

// Query Hugging Face Inference API
async function queryHuggingFaceAPI(model, data) {
    const response = await fetch(`${INFERENCE_API_URL}/${model}`, {
        headers: { 
            "Content-Type": "application/json" 
        },
        method: "POST",
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

// Convert image to base64
function getBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = error => reject(error);
    });
}

// Analyze image objects
async function analyzeImageObjects(imageFile) {
    try {
        const base64Image = await getBase64(imageFile);
        const results = await queryHuggingFaceAPI(OBJECT_DETECTION_MODEL, { 
            image: base64Image 
        });

        // Process and format detection results
        if (Array.isArray(results)) {
            const formattedResults = results.map(item => 
                `${item.label} (${(item.score * 100).toFixed(2)}% confidence)`
            ).join('\n');

            return formattedResults || 'No objects detected.';
        }

        return 'No objects detected.';
    } catch (error) {
        console.error('Image analysis error:', error);
        return 'Error analyzing image.';
    }
}

// Process image with object detection
async function processImage(imageFile) {
    try {
        const reader = new FileReader();
        return new Promise((resolve, reject) => {
            reader.onload = async function(e) {
                try {
                    const imageDescription = await analyzeImageObjects(imageFile);

                    resolve({
                        originalImage: e.target.result,
                        imageDescription: imageDescription
                    });
                } catch (error) {
                    reject(error);
                }
            };
            reader.readAsDataURL(imageFile);
        });
    } catch (error) {
        console.error('Image processing error:', error);
        alert('Error processing image.');
    }
}

function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) return;

    processImage(file).then(result => {
        // Create a note entry with image and analysis
        const noteEntry = document.createElement('div');
        noteEntry.classList.add('note-entry');
        noteEntry.innerHTML = `
            <img src="${result.originalImage}" alt="Uploaded image" class="uploaded-image">
            <div class="image-analysis">
                <h3>Image Understanding</h3>
                <pre>${result.imageDescription}</pre>
            </div>
        `;

        // Add to notes container
        notesContainer.appendChild(noteEntry);

        // Show processing section
        processingSection.classList.remove('hidden');
    }).catch(error => {
        console.error('Image upload error:', error);
        alert('Failed to process image. Please try again.');
    });
}

// Drag and drop listeners
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

dropzone.addEventListener('drop', handleDrop, false);
imageUpload.addEventListener('change', (e) => handleImageUpload(e.target.files[0]), false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleImageUpload(files[0]);
}

// Plain Text Export
plainTextBtn.addEventListener('click', () => {
    let plainText = '';
    const noteEntries = document.querySelectorAll('.note-entry');
    
    noteEntries.forEach((entry, index) => {
        const imageAnalysis = entry.querySelector('.image-analysis pre').textContent;
        plainText += `Note ${index + 1}:\n${imageAnalysis}\n\n`;
    });
    
    // Fallback for browsers that don't support clipboard API
    fallbackCopyTextToClipboard(plainText);
});

// Fallback clipboard function for broader browser support
function fallbackCopyTextToClipboard(text) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    
    try {
        const successful = document.execCommand('copy');
        alert(successful ? 'Copied to clipboard!' : 'Unable to copy');
    } catch (err) {
        alert('Unable to copy');
    }
    
    document.body.removeChild(textArea);
}