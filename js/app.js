// Hugging Face Model Configurations
const HUGGINGFACE_API_URL = 'https://api-inference.huggingface.co/models';

// Vision Transformer for Text Extraction
const VIT_OCR_MODEL = 'google/vit-base-patch16-224';

// Stable Diffusion for Visualization
const STABLE_DIFFUSION_MODEL = 'stabilityai/stable-diffusion-2-1';

// Key Models for Transformation
const MODELS = {
    textExtraction: 'google/vit-base-patch16-224',
    drawingToChart: 'stabilityai/stable-diffusion-2-1'
};

class NoteTransformer {
    constructor() {
        this.apiKey = null; // You'll need to add Hugging Face API key
    }

    async queryHuggingFace(model, data) {
        if (!this.apiKey) {
            throw new Error('Hugging Face API key is required');
        }

        const response = await fetch(`${HUGGINGFACE_API_URL}/${model}`, {
            headers: { 
                "Authorization": `Bearer ${this.apiKey}`,
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

    async extractTextFromImage(imageFile) {
        try {
            const base64Image = await this.getBase64(imageFile);
            const result = await this.queryHuggingFace(
                MODELS.textExtraction, 
                { image: base64Image }
            );

            return result.text || 'No text could be extracted.';
        } catch (error) {
            console.error('Text extraction error:', error);
            return 'Error extracting text.';
        }
    }

    async transformDrawingToVisualization(imageFile, textContext) {
        try {
            // Generate a prompt for Stable Diffusion based on extracted text
            const visualizationPrompt = this.generateVisualizationPrompt(textContext);
            
            const base64Image = await this.getBase64(imageFile);
            const result = await this.queryHuggingFace(
                MODELS.drawingToChart, 
                { 
                    inputs: visualizationPrompt,
                    image: base64Image 
                }
            );

            return result.generated_image || null;
        } catch (error) {
            console.error('Visualization transformation error:', error);
            return null;
        }
    }

    generateVisualizationPrompt(text) {
        // Intelligent prompt generation for Stable Diffusion
        const possiblePrompts = [
            `Create a professional data visualization chart representing: ${text}`,
            `Transform the context of: ${text} into an elegant infographic`,
            `Generate a clean, minimalist chart visualizing the key points: ${text}`
        ];

        // Randomly select a prompt style
        return possiblePrompts[Math.floor(Math.random() * possiblePrompts.length)];
    }

    async getBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.onerror = error => reject(error);
        });
    }

    async processWhiteboardImage(imageFile) {
        try {
            // Extract text first
            const extractedText = await this.extractTextFromImage(imageFile);
            
            // Transform drawing to visualization
            const visualizedImage = await this.transformDrawingToVisualization(imageFile, extractedText);

            return {
                originalImage: URL.createObjectURL(imageFile),
                extractedText,
                visualizedImage
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
                        <h3>Extracted Text</h3>
                        <pre>${result.extractedText}</pre>
                        
                        ${result.visualizedImage ? `
                        <h3>Visualization</h3>
                        <img src="${result.visualizedImage}" alt="Generated Visualization" class="visualization-image">
                        ` : ''}
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