class SketchAnalyzer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.drawing = false;
        this.currentTool = 'pencil';
        
        this.setupEventListeners();
        this.setupTools();
    }

    setupEventListeners() {
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));
    }

    setupTools() {
        const pencilBtn = document.querySelector('.tool-btn[data-tool="pencil"]');
        const eraserBtn = document.querySelector('.tool-btn[data-tool="eraser"]');
        const clearBtn = document.getElementById('clearBtn');
        const fileUpload = document.getElementById('fileUpload');

        pencilBtn.addEventListener('click', () => this.setTool('pencil'));
        eraserBtn.addEventListener('click', () => this.setTool('eraser'));
        clearBtn.addEventListener('click', () => this.clearCanvas());
        fileUpload.addEventListener('change', this.handleFileUpload.bind(this));
    }

    setTool(tool) {
        this.currentTool = tool;
        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.tool === tool) {
                btn.classList.add('active');
            }
        });

        this.ctx.strokeStyle = tool === 'pencil' ? '#000000' : '#FFFFFF';
        this.ctx.lineWidth = tool === 'pencil' ? 3 : 20;
    }

    startDrawing(e) {
        this.drawing = true;
        this.draw(e);
    }

    draw(e) {
        if (!this.drawing) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        this.ctx.lineTo(x, y);
        this.ctx.stroke();
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
    }

    stopDrawing() {
        this.drawing = false;
        this.ctx.beginPath();
    }

    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    getSketchData() {
        return this.canvas.toDataURL('image/png');
    }

    handleFileUpload(e) {
        const file = e.target.files[0];
        const reader = new FileReader();

        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
            };
            img.src = event.target.result;
        };

        reader.readAsDataURL(file);
    }
}

class ImageProcessor {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
    }

    processImage(imageData) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                // Resize and prepare image for analysis
                this.canvas.width = 224;
                this.canvas.height = 224;
                
                this.ctx.drawImage(img, 0, 0, 224, 224);
                
                // Extract image data
                const processedImageData = this.ctx.getImageData(0, 0, 224, 224);
                
                // Simple image processing techniques
                this.enhanceContrast(processedImageData);
                this.detectEdges(processedImageData);
                
                this.ctx.putImageData(processedImageData, 0, 0);
                
                resolve(this.canvas.toDataURL('image/png'));
            };
            img.onerror = reject;
            img.src = imageData;
        });
    }

    enhanceContrast(imageData) {
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
            const contrast = 1.5;
            
            data[i] = this.truncate(((data[i] - avg) * contrast) + avg);
            data[i + 1] = this.truncate(((data[i + 1] - avg) * contrast) + avg);
            data[i + 2] = this.truncate(((data[i + 2] - avg) * contrast) + avg);
        }
    }

    detectEdges(imageData) {
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;
        const newImageData = new ImageData(width, height);
        const newData = newImageData.data;

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                
                // Sobel operator for edge detection
                const gx = (
                    (-1 * this.getPixelValue(imageData, x-1, y-1, 0)) +
                    (1 * this.getPixelValue(imageData, x+1, y-1, 0)) +
                    (-2 * this.getPixelValue(imageData, x-1, y, 0)) +
                    (2 * this.getPixelValue(imageData, x+1, y, 0)) +
                    (-1 * this.getPixelValue(imageData, x-1, y+1, 0)) +
                    (1 * this.getPixelValue(imageData, x+1, y+1, 0))
                );

                const gy = (
                    (-1 * this.getPixelValue(imageData, x-1, y-1, 0)) +
                    (-2 * this.getPixelValue(imageData, x, y-1, 0)) +
                    (1 * this.getPixelValue(imageData, x-1, y+1, 0)) +
                    (2 * this.getPixelValue(imageData, x, y+1, 0))
                );

                const magnitude = Math.sqrt(gx * gx + gy * gy);
                const val = magnitude > 50 ? 255 : 0;

                newData[idx] = val;
                newData[idx + 1] = val;
                newData[idx + 2] = val;
                newData[idx + 3] = 255;
            }
        }

        // Copy processed data back
        for (let i = 0; i < data.length; i++) {
            data[i] = newData[i];
        }
    }

    getPixelValue(imageData, x, y, channel) {
        const idx = (y * imageData.width + x) * 4;
        return imageData.data[idx + channel];
    }

    truncate(value) {
        return Math.max(0, Math.min(255, value));
    }
}

class ChartGenerator {
    constructor() {
        this.colors = ['#000000', '#333333', '#666666', '#999999'];
    }

    generateChart(type, processingHint = null) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // Set up chart size
        canvas.width = 400;
        canvas.height = 400;

        // Generate chart based on the type
        if (type === 'bar') {
            this.generateBarChart(ctx, processingHint);
        } else if (type === 'line') {
            this.generateLineChart(ctx, processingHint);
        } else if (type === 'pie') {
            this.generatePieChart(ctx, processingHint);
        } else {
            console.error('Unsupported chart type:', type);
        }

        return canvas.toDataURL('image/png');
    }

    generateBarChart(ctx, processingHint) {
        const data = processingHint || [50, 60, 70, 80]; // Example data
        const barWidth = 50;
        const gap = 10;

        ctx.fillStyle = this.colors[0];
        data.forEach((value, index) => {
            ctx.fillRect(index * (barWidth + gap), 400 - value, barWidth, value);
        });
    }

    generateLineChart(ctx, processingHint) {
        const data = processingHint || [30, 50, 70, 90, 110]; // Example data
        ctx.beginPath();
        ctx.moveTo(0, 400 - data[0]);

        data.forEach((value, index) => {
            ctx.lineTo(index * 80, 400 - value);
        });

        ctx.strokeStyle = this.colors[1];
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    generatePieChart(ctx, processingHint) {
        const data = processingHint || [10, 30, 50, 40]; // Example data
        let total = data.reduce((acc, value) => acc + value, 0);
        let startAngle = 0;

        data.forEach((value, index) => {
            const endAngle = startAngle + (2 * Math.PI * (value / total));

            ctx.beginPath();
            ctx.moveTo(200, 200); // Center of the pie chart
            ctx.arc(200, 200, 100, startAngle, endAngle);
            ctx.fillStyle = this.colors[index % this.colors.length];
            ctx.fill();

            startAngle = endAngle;
        });
    }
}
