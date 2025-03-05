// Hugging Face Integration and Image Processing
const dropzone = document.getElementById('dropzone');
const imageUpload = document.getElementById('imageUpload');
const processingSection = document.getElementById('processing-section');
const originalImageWrapper = document.getElementById('original-image-wrapper');
const extractedTextPre = document.getElementById('extracted-text');
const imageDescriptionPre = document.getElementById('image-description');
const markdownBtn = document.getElementById('markdownBtn');
const jsonBtn = document.getElementById('jsonBtn');
const visualizeBtn = document.getElementById('visualizeBtn');
const visualizationSection = document.getElementById('visualization-section');
const visualizationContainer = document.getElementById('visualization-container');

// Hugging Face Inference API Endpoints
const INFERENCE_API_URL = 'https://api-inference.huggingface.co/models';

// Image to Text Model (Public)
const OCR_MODEL = 'microsoft/trocr-base-handwritten';

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

// Extract text from image
async function extractTextFromImage(imageFile) {
    try {
        const base64Image = await getBase64(imageFile);
        const result = await queryHuggingFaceAPI(OCR_MODEL, { 
            image: base64Image 
        });

        return result.generated_text || 'No text could be extracted.';
    } catch (error) {
        console.error('Text extraction error:', error);
        return 'Error extracting text.';
    }
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

// Process image with both text extraction and object detection
async function processImage(imageFile) {
    // Show loading state
    extractedTextPre.textContent = 'Extracting text...';
    imageDescriptionPre.textContent = 'Analyzing image...';

    try {
        const reader = new FileReader();
        return new Promise((resolve, reject) => {
            reader.onload = async function(e) {
                try {
                    const [extractedText, imageDescription] = await Promise.all([
                        extractTextFromImage(imageFile),
                        analyzeImageObjects(imageFile)
                    ]);

                    resolve({
                        originalImage: e.target.result,
                        extractedText: extractedText,
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
        extractedTextPre.textContent = 'Error processing image.';
        imageDescriptionPre.textContent = 'Error processing image.';
    }
}

function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) return;

    processImage(file).then(result => {
        // Show original image
        originalImageWrapper.innerHTML = `
            <img src="${result.originalImage}" alt="Uploaded image">
        `;

        // Update text extraction and image understanding
        extractedTextPre.textContent = result.extractedText;
        imageDescriptionPre.textContent = result.imageDescription;

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

// Conversion and visualization buttons
markdownBtn.addEventListener('click', () => {
    const extractedText = extractedTextPre.textContent;
    const markdownText = `## Extracted Text\n\n${extractedText}`;
    
    // Fallback for browsers that don't support clipboard API
    fallbackCopyTextToClipboard(markdownText);
});

jsonBtn.addEventListener('click', () => {
    const extractedText = extractedTextPre.textContent;
    const jsonText = JSON.stringify({
        extractedText: extractedText,
        timestamp: new Date().toISOString()
    }, null, 2);
    
    // Fallback for browsers that don't support clipboard API
    fallbackCopyTextToClipboard(jsonText);
});

visualizeBtn.addEventListener('click', () => {
    const uploadedImage = imageUpload.files[0];
    if (uploadedImage) {
        visualizationContainer.innerHTML = `
            <div class="visualization-placeholder">
                <h3>Visual Analysis Placeholder</h3>
                <p>Advanced visualization features coming soon!</p>
            </div>
        `;
        visualizationSection.classList.remove('hidden');
    } else {
        alert('Please upload an image first');
    }
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