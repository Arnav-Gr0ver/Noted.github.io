class NoteTransformer {
    constructor() {
        // Public, ungated models
        this.MODELS = {
            textExtraction: 'microsoft/trocr-base-handwritten',
            objectDetection: 'facebook/detr-resnet-50',
            imageClassification: 'google/vit-base-patch16-224'
        };
        
        // Hugging Face Inference API base URL
        this.INFERENCE_API_URL = 'https://api-inference.huggingface.co/models';
    }

    async queryHuggingFace(model, imageFile) {
        try {
            const formData = new FormData();
            formData.append('image', imageFile); // Send image as FormData

            const response = await fetch(`${this.INFERENCE_API_URL}/${model}`, {
                method: 'POST',
                body: formData // FormData instead of JSON
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Detailed Error Response:', errorText);
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Model Inference Error:', error);
            throw error;
        }
    }

    async extractTextFromImage(imageFile) {
        try {
            console.log('Attempting text extraction with model:', this.MODELS.textExtraction);
            
            const result = await this.queryHuggingFace(
                this.MODELS.textExtraction, 
                imageFile // Directly pass the image file
            );

            console.log('Text Extraction Result:', result);

            return result.generated_text || 
                   result.text || 
                   'No text could be extracted.';
        } catch (error) {
            console.error('Text extraction error:', error);
            return `Error extracting text: ${error.message}`;
        }
    }

    async analyzeImageObjects(imageFile) {
        try {
            console.log('Attempting object detection with model:', this.MODELS.objectDetection);
            
            const results = await this.queryHuggingFace(
                this.MODELS.objectDetection, 
                imageFile
            );

            console.log('Object Detection Results:', results);

            if (Array.isArray(results)) {
                return results.map(item => 
                    `${item.label} (${(item.score * 100).toFixed(2)}% confidence)`
                ).join('\n');
            }

            return 'No objects detected.';
        } catch (error) {
            console.error('Image analysis error:', error);
            return `Error analyzing image: ${error.message}`;
        }
    }

    async processWhiteboardImage(imageFile) {
        try {
            const [extractedText, imageDescription] = await Promise.all([
                this.extractTextFromImage(imageFile),
                this.analyzeImageObjects(imageFile)
            ]);

            return {
                originalImage: URL.createObjectURL(imageFile),
                extractedText,
                imageDescription
            };
        } catch (error) {
            console.error('Whiteboard image processing error:', error);
            return null;
        }
    }
}

// UI Interaction Setup
document.addEventListener('DOMContentLoaded', () => {
    const noteTransformer = new NoteTransformer();
    const dropzone = document.getElementById('dropzone');
    const imageUpload = document.getElementById('imageUpload');
    const processingSection = document.getElementById('processing-section');
    const notesContainer = document.getElementById('notes-container');

    function handleImageUpload(file) {
        if (!file.type.startsWith('image/')) return;

        noteTransformer.processWhiteboardImage(file).then(result => {
            if (result) {
                const noteEntry = document.createElement('div');
                noteEntry.classList.add('note-entry');
                noteEntry.innerHTML = `
                    <div class="original-image">
                        <img src="${result.originalImage}" alt="Original Whiteboard">
                    </div>
                    <div class="note-content">
                        <h3>Extracted Information</h3>
                        <div class="info-section">
                            <h4>Text Extraction</h4>
                            <pre>${result.extractedText}</pre>
                            
                            <h4>Image Understanding</h4>
                            <pre>${result.imageDescription}</pre>
                        </div>
                    </div>
                `;

                notesContainer.appendChild(noteEntry);
                processingSection.classList.remove('hidden');
            }
        }).catch(error => {
            console.error('Image processing error:', error);
            alert('Failed to process image. Please try again.');
        });
    }

    // Drag and drop event listeners
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        handleImageUpload(files[0]);
    }, false);

    imageUpload.addEventListener('change', (e) => {
        handleImageUpload(e.target.files[0]);
    }, false);
});
