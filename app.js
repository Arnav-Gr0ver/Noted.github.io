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

// Hugging Face API Configuration
const HUGGINGFACE_API_KEY = 'YOUR_HUGGING_FACE_API_KEY'; // Replace with your actual API key

// Simulated AI Processing (replace with actual API calls)
async function simulateTextExtraction(imageFile) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve('Simulated text extraction:\n\nThis is a sample of extracted text from the image.\nAn actual implementation would use Hugging Face OCR models.');
        }, 1000);
    });
}

async function simulateImageAnalysis(imageFile) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve('Image Analysis Results:\n\n- Detected objects: Various elements\n- Confidence: Simulated analysis\n- Complexity: Medium');
        }, 1000);
    });
}

async function processImage(imageFile) {
    return new Promise(async (resolve) => {
        const reader = new FileReader();
        reader.onload = async function(e) {
            const extractedText = await simulateTextExtraction(imageFile);
            const imageDescription = await simulateImageAnalysis(imageFile);
            
            resolve({
                originalImage: e.target.result,
                extractedText: extractedText,
                imageDescription: imageDescription
            });
        };
        reader.readAsDataURL(imageFile);
    });
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
                <h3>Visual Analysis</h3>
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